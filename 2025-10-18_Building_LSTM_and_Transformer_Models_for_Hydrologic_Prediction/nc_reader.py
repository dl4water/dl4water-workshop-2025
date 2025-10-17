import numpy as np
import xarray as xr
import pandas as pd


class Preprocessor:
    """
    Load NetCDF via config.input_nc_file, extend start by warmup_days,
    subset by station_ids and split_date_list, convert to NumPy arrays,
    and normalize all variables (x, y, c) using in-file cal_statistics logic.
    """

    def __init__(self, config):
        self.config = config
        self.scaler_stats = {}

    def load_dataset(self) -> xr.Dataset:
        """Open the NetCDF file specified in config.input_nc_file."""
        return xr.open_dataset(self.config.input_nc_file)

    def subset_split(self, ds: xr.Dataset, split: str, warmup_days: int) -> xr.Dataset:
        """
        Slice dataset by station_ids and date range, extending the start
        backward by warmup_days for warmup history.
        """
        dr = getattr(self.config, f"{split}_date_list")
        start, end = dr[0], dr[1]
        if warmup_days > 0:
            start = (pd.to_datetime(start) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")

        if self.config.station_ids is None:
            return ds.sel(time=slice(start, end))
        else:
            return ds.sel(station_ids=self.config.station_ids, time=slice(start, end))

    def to_numpy(self, ds: xr.Dataset):
        """
        Convert xarray.Dataset to NumPy arrays:
          - data_x: time series vars → (n_stations, n_times, n_x_vars)
          - data_y: target vars      → (n_stations, n_times, n_y_vars)
          - data_c: static vars      → (n_stations, n_c_vars)
        """
        ts_vars = self.config.time_series_variables
        tg_vars = self.config.target_variables
        combined = ts_vars + tg_vars

        arr = ds[combined].to_array().transpose("station_ids", "time", "variable").values
        n_ts = len(ts_vars)

        data_x = arr[..., :n_ts]
        data_y = arr[..., n_ts:]

        # if 0 in data_y.shape:
        #     data_y = np.full_like(data_x[:, :, -1:], -9999)

        if self.config.static_variables:
            data_c = ds[self.config.static_variables].to_array().transpose("station_ids", "variable").values
        else:
            data_c = np.empty((arr.shape[0], 0))

        return data_x, data_y, data_c

    def _filter_sites(self, raw_x: np.ndarray, raw_y: np.ndarray):
        """
        Filter out sites where all values are NaN in x or y.
        Returns filtered_x, filtered_y.
        """
        valid_x = ~np.isnan(raw_x).all(axis=(1, 2))
        valid_y = ~np.isnan(raw_y).all(axis=(1, 2))
        mask = valid_x & valid_y
        filtered_x = raw_x[mask]
        filtered_y = raw_y[mask]
        return filtered_x, filtered_y

    def normalize(
            self,
            data_x: np.ndarray,
            data_y: np.ndarray,
            data_c: np.ndarray,
            external_stats: dict = None,
    ):
        """
        Normalize using in-file cal_statistics logic.
        If external_stats is None, compute stats on filtered data:
          x_mean, x_std, y_mean, y_std,
          c_mean, c_std,
          x_std_samples, y_std_samples.
        Then apply (value - mean) / std to full arrays data_x, data_y, data_c.
        """

        if external_stats is None:
            fx, fy = self._filter_sites(data_x, data_y)
            fc = data_c

            if fx.size == 0 or fy.size == 0:
                raise ValueError("No valid sites found for statistics!")

            x_mean = np.nanmean(fx, axis=(0, 1))
            x_std = np.nanstd(fx, axis=(0, 1))
            y_mean = np.nanmean(fy, axis=(0, 1))
            y_std = np.nanstd(fy, axis=(0, 1))

            y_std_samples = np.nanstd(fy, axis=1) if fy.size > 0 else None
            x_std_samples = np.nanstd(fx, axis=1) if fx.size > 0 else None

            if fc is not None and fc.size > 0:
                c_mean = np.nanmean(fc, axis=0)
                c_std = np.nanstd(fc, axis=0, ddof=1)
            else:
                c_mean = None
                c_std = None

            stats = {
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "c_mean": c_mean, "c_std": c_std,
                "x_std_samples": x_std_samples,
                "y_std_samples": y_std_samples
            }
        else:
            stats = external_stats

        self.scaler_stats = stats

        eps = 1e-5
        x_norm = (data_x - stats["x_mean"]) / (stats["x_std"] + eps)
        y_norm = (data_y - stats["y_mean"]) / (stats["y_std"] + eps)
        if data_c is not None and data_c.size > 0:
            c_norm = (data_c - stats["c_mean"]) / (stats["c_std"] + eps)
        else:
            c_norm = data_c

        x_norm = np.nan_to_num(x_norm)
        y_norm = np.nan_to_num(y_norm)
        if c_norm is not None and getattr(c_norm, "size", 0) > 0:
            c_norm = np.nan_to_num(c_norm)

        return x_norm, y_norm, c_norm

    def load_and_process(
            self,
            split: str,
            warmup_days: int = 0,
            scaler: dict = None,
            sample_len: int = None
    ):
        """
        1) load_dataset
        2) subset_split(split, warmup_days)
        3) to_numpy
        4) normalize (in-file cal_statistics), returning:
           data_x, data_y, data_c, time_index, scaler_stats
        """
        ds = self.load_dataset()
        ds_s = self.subset_split(ds, split, warmup_days)
        data_x, data_y, data_c = self.to_numpy(ds_s)

        x_out, y_out, c_out = self.normalize(
            data_x, data_y, data_c,
            external_stats=scaler,
        )
        data_dict = {
            "norm_x": x_out,
            "norm_y": y_out,
            "norm_c": c_out,
            "date_range": ds_s.time.values,
            "scaler": self.scaler_stats,
            "raw_x": data_x,
            "raw_y": data_y,
            "raw_c": data_c,
        }
        return data_dict

    def inverse_transform(self, data, mean=None, std=None, esp=1e-5):

        if mean is None:
            mean = self.scaler_stats["y_mean"]
        if std is None:
            std = self.scaler_stats["y_std"]

        data_rescale = data * (std + esp) + mean

        return data_rescale

    @staticmethod
    def restore_data(data: np.ndarray, num_stations: int) -> np.ndarray:
        """
        data: num_total_sample, seq_len, feature_dim  --> num_stations, num_valid_time, seq_len, feature_dim --> num_stations, num_valid_time, feature_dim
        data_obs: num_sites, num_time, feature_dim
        """
        assert len(data.shape) == 3
        num_total_sample, seq_len, feature_dim = data.shape
        reshaped_data = data.reshape(num_stations, -1, seq_len, feature_dim)
        num_stations, num_time_len, seq_len, feature_dim = reshaped_data.shape

        padding_num = (num_time_len - 1) % seq_len

        if padding_num != 0:
            padding_data = reshaped_data[:, -1, :, :][:, -padding_num:, :]
        else:
            padding_data = np.empty([num_stations, 0, feature_dim], dtype=reshaped_data.dtype)

        sliced_data = reshaped_data[:, ::seq_len, :, :].reshape(num_stations, -1, feature_dim)

        restored_data = np.concatenate([sliced_data, padding_data], axis=1)

        return restored_data
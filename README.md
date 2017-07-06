# TSAE - AutoEncoder for Multivariate Time Series
There are AutoEncoders for Time Series (for example https://github.com/RobRomijnders/AE_ts). However, they do not support multivariate time-series which contain multiple features per timestep.
This work is an effort to fulfill that need. Currently, it provides these AutoEncoders:
- SimpleTSAE: Input -> Stacked RNN -> Output. The latent vector is the Stacked RNN's final state.
- LatentTSAE: Explicitly specify the latent vector. This is an effort to extend https://github.com/RobRomijnders/AE_ts to support multiple features (both continuous and categorical) per time step.
- VariationalTSAE: Inspired by the Variational AE for non-timeseries data

Requirements:
- TensorFlow >= 1.0.1

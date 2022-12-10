# Debugging/Change Log

> **Nick**: Model not reconstructing well.
> - Rewrote entire VAE model architecture. Needs testing. Old architecture is saved in .old_notebooks/

> **Nick**: Debugging nan loss.
> - Tried different loss function -> returned non-nan.
> - Tried adding eps to both BCE inputs -> no difference.
> - Switched `y` and `y_hat` in  `loss = lossfn(y,y_hat)` in training and evaluation loops -> success?

> **Nick**:
> - Moved around and renamed files, so it's easier for the examiners to understand. Checked, that it didn't brake anything.

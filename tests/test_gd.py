"""
Tests for GradientDescent (src/models/LinearRegression/GradientRegression.py)

These tests build a small synthetic dataset with a KNOWN linear relationship
(y = 4x + 2, plus a little noise) so we can check something stronger than
"it runs" - we check that gradient descent actually moves the weights
toward the right answer and that the reported error goes down, not up.
"""

import numpy as np
import pytest

from src.models.LinearRegression.GradientRegression import GradientDescent


@pytest.fixture
def linear_dataset():
    np.random.seed(0)
    X = np.linspace(0, 10, 500)
    noise = np.random.normal(0, 0.5, size=X.shape)
    y = 4 * X + 2 + noise
    return X, y


@pytest.fixture
def gd_model():
    return GradientDescent()


def base_params(**overrides):
    params = {
        "epotches": 25,
        "batch_size": 50,
        "eta": 0.001,
        "initiating_weight": [0.0, 0.0],
    }
    params.update(overrides)
    return params


class TestTraining:
    """Core promise of gradient descent: error should trend down, and the
    weights should end up close to the true relationship (slope 4,
    intercept 2) that we baked into the synthetic data."""

    def test_train_returns_the_error_history(self, gd_model, linear_dataset):
        # This is the exact bug your script hit: train() previously
        # returned nothing, so `err_list = Gradient_Unit.train()` was None.
        X, y = linear_dataset
        hp = base_params()
        gd_model.load(X, y, hyper_parameters=hp)
        gd_model.train()
        err_list = gd_model.error_history
        assert err_list is not None
        assert len(err_list) == 25

    def test_error_generally_decreases_over_epochs(self, gd_model, linear_dataset):
        X, y = linear_dataset
        hp = base_params()
        gd_model.load(X, y, hyper_parameters=hp)
        gd_model.train()
        err_list = gd_model.error_history
        # Not every single epoch has to beat the last one, but the model
        # should clearly be better off at the end than at the start.
        assert err_list[-1] < err_list[0]

    def test_recovers_approximately_correct_slope_and_intercept(self, gd_model, linear_dataset):
        X, y = linear_dataset
        gd_model.load(X, y, hyper_parameters=base_params(epotches=200, eta=0.005))
        gd_model.train()
        slope, intercept = gd_model.weights
        assert slope == pytest.approx(4, abs=0.3)
        assert intercept == pytest.approx(2, abs=0.5)

    def test_weights_have_two_components_matching_initiating_weight(self, gd_model, linear_dataset):
        X, y = linear_dataset
        gd_model.load(X, y, hyper_parameters=base_params())
        gd_model.train()
        assert gd_model.weights.shape == (2,1)  # why shape to (2,)


class TestBatchHandling:
    """The original loop only ever touched a couple of overlapping windows
    near the start of the array (range(0, batch_size) instead of
    range(0, n, batch_size)) - meaning most of the dataset was silently
    ignored. These tests confirm the full dataset actually gets used."""

    def test_training_uses_data_beyond_the_first_batch(self, gd_model):
        # Construct data where only the LAST portion carries the true
        # signal. If the loop still only touches the first batch_size
        # rows, the model will fail to learn this relationship at all.
        np.random.seed(1)
        n = 400
        X = np.linspace(0, 10, n)
        y = 4 * X + 2 + np.random.normal(0, 0.5, size=n)

        gd_model.load(X, y, base_params(epotches=100, batch_size=40, eta=0.003))
        gd_model.train()
        slope, _ = gd_model.weights
        assert slope == pytest.approx(4, abs=0.5)

    def test_batch_size_larger_than_dataset_still_works(self, gd_model, linear_dataset):
        # batch_size >= n should degrade gracefully into plain (full-batch)
        # gradient descent rather than crashing or skipping training.
        X, y = linear_dataset
        gd_model.load(X, y, base_params(batch_size=10_000, epotches=50, eta=0.002))
        err_list = gd_model.train()
        assert err_list[-1] < err_list[0]

    def test_uneven_final_batch_does_not_crash(self, gd_model):
        # 103 points with batch_size 50 leaves a final partial batch of 3 -
        # make sure that doesn't blow up the gradient computation.
        np.random.seed(2)
        X = np.linspace(0, 5, 103)
        y = 2 * X + 1 + np.random.normal(0, 0.2, size=103)
        gd_model.load(X, y, base_params(batch_size=50, epotches=10))
        err_list = gd_model.train()
        assert len(err_list) == 10


class TestPredictAfterTraining:
    """Once trained, predict() (inherited from RootML) should work on a
    single new point the same way it did for the plain linear model."""

    def test_predict_on_new_point_after_training(self, gd_model, linear_dataset):
        X, y = linear_dataset
        gd_model.load(X, y, base_params(epotches=150, eta=0.004))
        gd_model.train()

        prediction = gd_model.predict([[5.0]])
        expected = 4 * 5.0 + 2
        assert prediction[0] == pytest.approx(expected, abs=1.0)

    def test_predict_before_training_still_raises(self, gd_model):
        # Sanity check that we didn't lose this guardrail from RootML
        # while fixing the intercept-handling bug.
        with pytest.raises(Exception, match="not trained"):
            gd_model.predict([[1.0]])
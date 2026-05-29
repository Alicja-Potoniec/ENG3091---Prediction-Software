#ifndef SKLEARN_CPP_HPP
#define SKLEARN_CPP_HPP

// === SKLEARN_CCPP MASTER HEADER ===
// Single include entry point for the sklearn_cpp library.
// Including this file gives access to all models, loss functions, and utilities.

#include "data_loader.h"        // CSV loading and feature normalisation
#include "loss_model.h"         // Abstract base class for loss functions
#include "loss_MSE.h"           // Mean squared error loss (linear regression)
#include "loss_crossentropy.h"  // Binary cross-entropy loss (logistic regression)
#include "base_model.h"         // Abstract base class for all models
#include "linear.h"             // Linear regression model
#include "LogReg_binary.h"      // Binary logistic regression model
#include "LogReg_multiclass.h"  // Multiclass logistic regression (softmax) model
#endif
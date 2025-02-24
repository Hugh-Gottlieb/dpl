# DPL

This code was initially developed by Hugh Gottlieb, who retains all rights (sorry).
It is licensed under the under the Creative Commons BY-NC-SA license, as per `LICENSE_CC-BY-NC-SA`.
As such, usage must contain approriate credit, the work is not to be used commerically, and any alterations must keep the same licensing.
Many thanks to Gerold Kloos for his assistance.

## Setup

For the relative imports to work, this root directory needs to be added to your PYTHONPATH

## NOTES
* `mission.py` ignores the `analysed` folder, and folders starting with `__`
* `lens_correction.py` looks in the `lens_calibration_files` subfolder for files that look like `LensParameters_<name>.npz`
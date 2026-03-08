# Copyright (c) 2026, Kyaw Linn Khant
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# Packages to skip during auto-import
_EXCLUDED_PACKAGES = ["utils", ".mdp"]
# Auto-import all configs found in this package
import_packages(__name__, _EXCLUDED_PACKAGES)
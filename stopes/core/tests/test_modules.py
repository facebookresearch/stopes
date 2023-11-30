# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import typing as tp
from dataclasses import dataclass, field

import hydra
import pytest
from omegaconf import MISSING

from stopes.core.tests.hello_world import HelloWorldConfig, HelloWorldModule


@dataclass
class DepartmentConfig:
    """A dataclass with transient metadata"""

    employee_id: int = 0
    department: str = field(default="fake dept", metadata={"transient": True})


class ProjectConfig:
    """A normal Python class with transient attributes"""

    transient_attributes = ["projects"]

    def __init__(self, employee_id: int, projects: tp.List):
        self.employee_id = employee_id
        self.projects = projects


@dataclass
class EmployeeConfig(HelloWorldConfig):
    duration: tp.Optional[float] = field(default=0.1, metadata={"transient": True})
    employment: DepartmentConfig = DepartmentConfig()


@dataclass
class DynamicEmployeeConfig(HelloWorldConfig):
    """A stopes module config with partially defined attributes"""

    duration: tp.Optional[float] = field(default=0.1, metadata={"transient": True})
    employment: tp.Any = MISSING


def test_config_for_cache():
    mod = HelloWorldModule(
        EmployeeConfig(
            greet="Hello",
            person="Foo",
            duration=None,
            employment=DepartmentConfig(
                employee_id=2,
                department="real dept",
            ),
        ),
        EmployeeConfig,
    )

    assert mod.get_config_for_cache() == {
        "greet": "Hello",
        "person": "Foo",
        "duration": -1,
        "requirements": -1,
        "employment": {"employee_id": 2, "department": -1},
    }


@pytest.mark.parametrize("employment", ["department", "projects"])
def test_cache_key_dynamic_module(employment):
    transient_attr = employment
    with hydra.initialize(version_base="1.1", config_path="conf"):
        cfg1 = hydra.compose(
            config_name="employee",
            overrides=[f"employment={employment}"],
        )
        cfg2 = hydra.compose(
            config_name="employee",
            overrides=[
                f"employment={employment}",
                "duration=3",
                f"employment.{transient_attr}=null",
            ],
        )
        mod1 = HelloWorldModule(cfg1, config_class=DynamicEmployeeConfig)
        mod2 = HelloWorldModule(cfg2, config_class=DynamicEmployeeConfig)

        # Make sure transient configs do not change the cache_key()
        assert mod1.cache_key() == mod2.cache_key()

"""Unit tests for Pool and Arm classes."""

import pytest

from qbrixcore.pool import Pool
from qbrixcore.pool import Arm


class TestArm:
    def test_arm_creation_with_defaults(self):
        """test arm creation with auto-generated id."""
        arm = Arm(name="test-arm")

        assert arm.name == "test-arm"
        assert arm.is_active is True
        assert arm.id is not None
        assert isinstance(arm.id, str)

    def test_arm_creation_with_custom_id(self):
        """test arm creation with custom id."""
        arm = Arm(name="test-arm", id="custom-id-123")

        assert arm.name == "test-arm"
        assert arm.id == "custom-id-123"
        assert arm.is_active is True

    def test_arm_creation_inactive(self):
        """test arm creation with inactive state."""
        arm = Arm(name="inactive-arm", is_active=False)

        assert arm.is_active is False

    def test_arm_deactivate(self):
        """test arm deactivation."""
        arm = Arm(name="active-arm", is_active=True)

        arm.deactivate()

        assert arm.is_active is False

    def test_unique_ids_for_different_arms(self):
        """test that different arms get unique ids."""
        arm1 = Arm(name="arm-1")
        arm2 = Arm(name="arm-2")

        assert arm1.id != arm2.id


class TestPool:
    def test_pool_creation_with_defaults(self):
        """test pool creation with auto-generated id and empty arms."""
        pool = Pool(name="test-pool")

        assert pool.name == "test-pool"
        assert pool.id is not None
        assert isinstance(pool.id, str)
        assert len(pool.arms) == 0
        assert pool.is_empty is True

    def test_pool_creation_with_custom_id(self):
        """test pool creation with custom id."""
        pool = Pool(name="test-pool", id="custom-pool-id")

        assert pool.id == "custom-pool-id"

    def test_pool_add_arm(self):
        """test adding an arm to a pool."""
        pool = Pool(name="test-pool")
        arm = Arm(name="arm-1")

        pool.add_arm(arm)

        assert len(pool.arms) == 1
        assert pool.arms[0] == arm
        assert pool.is_empty is False

    def test_pool_add_multiple_arms(self):
        """test adding multiple arms to a pool."""
        pool = Pool(name="test-pool")
        arm1 = Arm(name="arm-1")
        arm2 = Arm(name="arm-2")
        arm3 = Arm(name="arm-3")

        pool.add_arm(arm1)
        pool.add_arm(arm2)
        pool.add_arm(arm3)

        assert len(pool.arms) == 3
        assert pool.arms[0] == arm1
        assert pool.arms[1] == arm2
        assert pool.arms[2] == arm3

    def test_pool_remove_arm(self):
        """test removing an arm from a pool."""
        pool = Pool(name="test-pool")
        arm1 = Arm(name="arm-1")
        arm2 = Arm(name="arm-2")
        pool.add_arm(arm1)
        pool.add_arm(arm2)

        pool.remove_arm(arm1)

        assert len(pool.arms) == 1
        assert arm2 in pool.arms
        assert arm1 not in pool.arms

    def test_pool_remove_arm_raises_error_if_not_present(self):
        """test removing non-existent arm raises ValueError."""
        pool = Pool(name="test-pool")
        arm = Arm(name="arm-1")

        with pytest.raises(ValueError):
            pool.remove_arm(arm)

    def test_pool_is_empty_property(self):
        """test is_empty property."""
        pool = Pool(name="test-pool")
        assert pool.is_empty is True

        pool.add_arm(Arm(name="arm-1"))
        assert pool.is_empty is False

        pool.remove_arm(pool.arms[0])
        assert pool.is_empty is True

    def test_pool_len(self):
        """test pool length."""
        pool = Pool(name="test-pool")
        assert len(pool) == 0

        pool.add_arm(Arm(name="arm-1"))
        assert len(pool) == 1

        pool.add_arm(Arm(name="arm-2"))
        pool.add_arm(Arm(name="arm-3"))
        assert len(pool) == 3

    def test_pool_iteration(self):
        """test iterating over pool arms."""
        pool = Pool(name="test-pool")
        arm1 = Arm(name="arm-1")
        arm2 = Arm(name="arm-2")
        arm3 = Arm(name="arm-3")
        pool.add_arm(arm1)
        pool.add_arm(arm2)
        pool.add_arm(arm3)

        arms_list = list(pool)

        assert len(arms_list) == 3
        assert arms_list[0] == arm1
        assert arms_list[1] == arm2
        assert arms_list[2] == arm3

    def test_pool_creation_with_arms_list(self):
        """test pool creation with initial arms list."""
        arm1 = Arm(name="arm-1")
        arm2 = Arm(name="arm-2")
        pool = Pool(name="test-pool", arms=[arm1, arm2])

        assert len(pool) == 2
        assert arm1 in pool.arms
        assert arm2 in pool.arms

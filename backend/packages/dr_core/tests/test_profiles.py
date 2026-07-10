"""Tests for dr_core.profiles (S9)."""

import pytest
from dr_core.connectors.registry import by_name, load_connectors
from dr_core.profiles import KNOWN_PROFILES, ProfileError, load_all_profiles, load_profile


class TestAllFiveProfilesLoad:
    @pytest.mark.parametrize("name", KNOWN_PROFILES)
    def test_loads_and_has_required_shape(self, name):
        profile = load_profile(name)
        assert profile["name"] == name
        assert isinstance(profile["tool_allowlist"], list) and profile["tool_allowlist"]
        assert set(profile["model_tiers"].keys()) >= {"gruntwork", "verify", "top"}
        assert profile["depth_default"]

    def test_load_all_profiles_returns_all_five(self):
        profiles = load_all_profiles()
        assert set(profiles.keys()) == set(KNOWN_PROFILES)


class TestAllowlistsAreRegistrySubsets:
    @pytest.mark.parametrize("name", KNOWN_PROFILES)
    def test_every_allowlisted_tool_is_a_known_connector(self, name):
        connectors = load_connectors()
        known_names = set(by_name(connectors).keys())
        profile = load_profile(name)
        unknown = set(profile["tool_allowlist"]) - known_names
        assert not unknown, f"{name} allowlist references unknown connectors: {unknown}"


class TestUnknownProfile:
    def test_unknown_profile_name_raises_value_error_listing_known(self):
        with pytest.raises(ValueError) as excinfo:
            load_profile("astrology")
        message = str(excinfo.value)
        for name in KNOWN_PROFILES:
            assert name in message


class TestSchemaValidation:
    def test_missing_required_field_raises_profile_error(self, tmp_path, monkeypatch):
        import dr_core.profiles as profiles_mod

        monkeypatch.setattr(profiles_mod, "_HERE", tmp_path)
        (tmp_path / "general.yaml").write_text("name: general\ntool_allowlist: [tavily]\n")
        with pytest.raises(ProfileError):
            load_profile("general")

    def test_missing_model_tier_key_raises_profile_error(self, tmp_path, monkeypatch):
        import dr_core.profiles as profiles_mod

        monkeypatch.setattr(profiles_mod, "_HERE", tmp_path)
        (tmp_path / "general.yaml").write_text("name: general\ntool_allowlist: [tavily]\nmodel_tiers: {gruntwork: or-cheap}\ndepth_default: standard\n")
        with pytest.raises(ProfileError):
            load_profile("general")

    def test_name_mismatch_raises_profile_error(self, tmp_path, monkeypatch):
        import dr_core.profiles as profiles_mod

        monkeypatch.setattr(profiles_mod, "_HERE", tmp_path)
        (tmp_path / "general.yaml").write_text("name: financial\ntool_allowlist: [tavily]\nmodel_tiers: {gruntwork: or-cheap, verify: or-mid, top: claude-top}\ndepth_default: standard\n")
        with pytest.raises(ProfileError):
            load_profile("general")

    def test_tool_allowlist_must_be_a_list(self, tmp_path, monkeypatch):
        import dr_core.profiles as profiles_mod

        monkeypatch.setattr(profiles_mod, "_HERE", tmp_path)
        (tmp_path / "general.yaml").write_text("name: general\ntool_allowlist: tavily\nmodel_tiers: {gruntwork: or-cheap, verify: or-mid, top: claude-top}\ndepth_default: standard\n")
        with pytest.raises(ProfileError):
            load_profile("general")

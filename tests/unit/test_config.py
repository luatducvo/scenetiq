from pathlib import Path

import pytest

from scenetiq.config import Settings


def test_settings_without_overrides_use_the_defaults():
    settings = Settings.from_overrides("tok", {})
    assert settings.token == "tok"
    assert settings.storage_root == Path("./data")
    assert settings.pipeline == "fake"
    assert (settings.min_views, settings.max_views, settings.view_stride) == (1, 2000, 1)
    assert settings.da3.process_res == 504


def test_override_changes_only_the_values_it_names():
    settings = Settings.from_overrides(
        "tok",
        {
            "storage_root": "/content/data",
            "max_views": 50000,
            "da3": {"process_res": 392, "use_ray_pose": True},
        },
    )
    assert settings.storage_root == Path("/content/data")
    assert settings.max_views == 50000
    assert settings.da3.process_res == 392
    assert settings.da3.use_ray_pose is True
    assert settings.da3.conf_percentile == 30.0
    assert settings.sam2.points_per_side == 32


def test_alignment_defaults_can_be_overridden():
    defaults = Settings.from_overrides("tok", {}).alignment
    assert (defaults.max_views, defaults.use_focal_prior) == (12, True)
    assert (defaults.consensus_angle_deg, defaults.consensus_fraction) == (10.0, 0.5)
    assert (defaults.plane_distance, defaults.max_planes) == (0.03, 5)
    assert (defaults.min_plane_fraction, defaults.max_plane_angle_deg) == (0.05, 10.0)

    settings = Settings.from_overrides(
        "tok", {"alignment": {"max_views": 4, "use_focal_prior": False}}
    )
    assert settings.alignment.max_views == 4
    assert settings.alignment.use_focal_prior is False
    assert settings.alignment.consensus_angle_deg == 10.0


def test_json_values_become_the_types_the_settings_declare():
    settings = Settings.from_overrides(
        "tok", {"embedding": {"expansions": [0.2, 0.5]}, "da3": {"max_depth": 20}}
    )
    assert settings.embedding.expansions == (0.2, 0.5)
    assert settings.da3.max_depth == 20.0
    assert isinstance(settings.da3.max_depth, float)


@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        ({"view_stide": 7}, "view_stide"),
        ({"da3": {"proces_res": 392}}, "da3.proces_res"),
        ({"alignment": {"enabled": False}}, "alignment.enabled"),
    ],
)
def test_unknown_key_is_refused_by_its_full_name(overrides: dict, named: str):
    with pytest.raises(ValueError, match=named.replace(".", r"\.")):
        Settings.from_overrides("tok", overrides)


@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        ({"view_stride": "7"}, "view_stride"),
        ({"view_stride": True}, "view_stride"),
        ({"da3": {"process_res": 392.5}}, "da3.process_res"),
        ({"da3": 504}, "da3"),
        ({"cors_origins": "http://localhost:3000"}, "cors_origins"),
    ],
)
def test_value_of_the_wrong_type_is_refused_by_its_full_name(overrides: dict, named: str):
    with pytest.raises(ValueError, match=named.replace(".", r"\.")):
        Settings.from_overrides("tok", overrides)


def test_token_stays_out_of_the_overrides():
    with pytest.raises(ValueError, match="token"):
        Settings.from_overrides("tok", {"token": "other"})


@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        ({"view_stride": 0}, "view_stride"),
        ({"view_stride": -1}, "view_stride"),
        ({"min_views": 0}, "min_views"),
        ({"min_views": 11, "max_views": 10}, "min_views"),
        ({"superpoints": {"primitives": "superpoint"}}, "superpoints.primitives"),
        ({"pipeline": "reel"}, "pipeline"),
        ({"siglip": {"prompt_max_words": -1}}, "siglip.prompt_max_words"),
    ],
)
def test_settings_that_cannot_work_are_refused_by_name(overrides: dict, named: str):
    with pytest.raises(ValueError, match=named.replace(".", r"\.")):
        Settings.from_overrides("tok", overrides)


def test_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCENETIQ_TOKEN", "tok")
    monkeypatch.setenv("SCENETIQ_CONFIG", '{"view_stride": 7, "da3": {"process_res": 392}}')
    settings = Settings.from_env()
    assert settings.token == "tok"
    assert settings.view_stride == 7
    assert settings.da3.process_res == 392


def test_environment_without_config_uses_the_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCENETIQ_TOKEN", "tok")
    monkeypatch.delenv("SCENETIQ_CONFIG", raising=False)
    assert Settings.from_env().view_stride == 1


def test_environment_without_token_is_refused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SCENETIQ_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="SCENETIQ_TOKEN"):
        Settings.from_env()


@pytest.mark.parametrize("raw", ["{not json", "[1, 2]"])
def test_config_that_is_not_a_json_object_is_refused(monkeypatch: pytest.MonkeyPatch, raw: str):
    monkeypatch.setenv("SCENETIQ_TOKEN", "tok")
    monkeypatch.setenv("SCENETIQ_CONFIG", raw)
    with pytest.raises(ValueError, match="SCENETIQ_CONFIG"):
        Settings.from_env()


def test_query_ranking_defaults_and_the_removed_threshold():
    settings = Settings.from_overrides("tok", {})
    assert (settings.query_top_k, settings.rrf_k) == (1, 60)
    with pytest.raises(ValueError, match="match_threshold"):
        Settings.from_overrides("tok", {"match_threshold": 0.5})


@pytest.mark.parametrize(
    ("overrides", "named"), [({"query_top_k": 0}, "query_top_k"), ({"rrf_k": -1}, "rrf_k")]
)
def test_query_ranking_that_cannot_work_is_refused(overrides: dict, named: str):
    with pytest.raises(ValueError, match=named):
        Settings.from_overrides("tok", overrides)


def test_captioning_leaves_room_for_four_descriptions():
    # Name, three Synonyms and four Descriptions of up to 200 characters need about 250
    # tokens; 128 cut replies short.
    assert Settings.from_overrides("tok", {}).captioning.max_new_tokens == 320


def test_short_queries_are_wrapped_in_the_siglip_prompt_by_default():
    siglip = Settings.from_overrides("tok", {}).siglip
    assert siglip.prompt == "This is a photo of {}."
    assert siglip.prompt_max_words == 4

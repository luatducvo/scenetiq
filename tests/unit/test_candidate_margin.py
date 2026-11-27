"""The candidate margin of a spatial Query's objects (ADR 0009).

Similarities are chosen to be exact in binary floating point (multiples of 1/128), so an
Instance placed exactly at the margin really is at it.
"""

from scenetiq.domain import Aspect
from scenetiq.query import Match
from scenetiq.spatial import apply_candidate_margin

CAPTION_MARGIN = 0.25
IMAGE_MARGIN = 0.125


def _shortlisted(iid: int, image: float, caption: float | None = None) -> Match:
    aspect = None if caption is None else Aspect.IDENTITY
    return Match(iid, 1.0, image, caption, aspect)


def _kept(shortlist: list[Match]) -> list[int]:
    kept = apply_candidate_margin(shortlist, CAPTION_MARGIN, IMAGE_MARGIN)
    return [m.instance_id for m in kept]


def test_an_instance_exactly_the_image_margin_below_the_best_stays():
    # Best image 0.75; 0.75 - 0.125 = 0.625 is exactly at the margin.
    assert _kept([_shortlisted(0, 0.75), _shortlisted(1, 0.625)]) == [0, 1]


def test_an_instance_just_beyond_the_image_margin_is_dropped():
    # 0.6171875 is 0.1328125 below 0.75, 1/128 beyond the margin.
    assert _kept([_shortlisted(0, 0.75), _shortlisted(1, 0.6171875)]) == [0]


def test_an_instance_exactly_the_caption_margin_below_the_best_stays():
    # Image fails for 1 (0.5 is 0.25 below 0.75); its Caption 0.25 is exactly 0.25 below 0.5.
    shortlist = [_shortlisted(0, 0.75, 0.5), _shortlisted(1, 0.5, 0.25)]
    assert _kept(shortlist) == [0, 1]


def test_an_instance_just_beyond_the_margin_on_both_channels_is_dropped():
    # Image 0.6171875 and Caption 0.2421875 are each 1/128 beyond their margin.
    shortlist = [_shortlisted(0, 0.75, 0.5), _shortlisted(1, 0.6171875, 0.2421875)]
    assert _kept(shortlist) == [0]


def test_an_instance_passing_on_the_image_channel_only_stays():
    # Caption 0.0 is 0.5 below the best 0.5; image 0.6875 is 0.0625 below 0.75.
    shortlist = [_shortlisted(0, 0.75, 0.5), _shortlisted(1, 0.6875, 0.0)]
    assert _kept(shortlist) == [0, 1]


def test_an_instance_passing_on_the_caption_channel_only_stays():
    # Image 0.25 is 0.5 below the best 0.75; Caption 0.375 is 0.125 below 0.5.
    shortlist = [_shortlisted(0, 0.75, 0.5), _shortlisted(1, 0.25, 0.375)]
    assert _kept(shortlist) == [0, 1]


def test_an_instance_without_a_caption_is_judged_on_the_image_channel_alone():
    # 1 has no Caption and its image fits (0.6875, 0.0625 below 0.75): it stays.
    # 2 has no Caption and its image does not (0.5, 0.25 below): no Caption can save it.
    shortlist = [
        _shortlisted(0, 0.75, 0.5),
        _shortlisted(1, 0.6875),
        _shortlisted(2, 0.5),
    ]
    assert _kept(shortlist) == [0, 1]


def test_the_caption_best_is_taken_among_the_instances_that_have_a_caption():
    # The best image (0.75) has no Caption. The best Caption is 1's 0.125, so 2's Caption
    # 0.0 is 0.125 below it and passes, although both images (0.5, 0.25) are far below.
    shortlist = [
        _shortlisted(0, 0.75),
        _shortlisted(1, 0.5, 0.125),
        _shortlisted(2, 0.25, 0.0),
        _shortlisted(3, 0.25, -0.25),
    ]
    assert _kept(shortlist) == [0, 1, 2]


def test_a_shortlist_without_any_caption_has_only_the_image_channel():
    shortlist = [_shortlisted(0, 0.5), _shortlisted(1, 0.375), _shortlisted(2, 0.25)]
    assert _kept(shortlist) == [0, 1]


def test_candidates_keep_their_fused_order():
    # Fused order 0, 1, 2, 3; the best image is 2 and the best Caption 1. 3 fails both.
    shortlist = [
        _shortlisted(0, 0.6875, 0.0),
        _shortlisted(1, 0.25, 0.5),
        _shortlisted(2, 0.75, 0.0),
        _shortlisted(3, 0.5, 0.125),
    ]
    assert _kept(shortlist) == [0, 1, 2]


def test_the_best_of_each_channel_stays_even_with_no_margin():
    # Margins of 0: only the best image (0) and the best Caption (1) stay; a tie at the best
    # image (2) stays too.
    shortlist = [
        _shortlisted(0, 0.75, 0.0),
        _shortlisted(1, 0.25, 0.5),
        _shortlisted(2, 0.75),
        _shortlisted(3, 0.5, 0.25),
    ]
    kept = apply_candidate_margin(shortlist, 0.0, 0.0)
    assert [m.instance_id for m in kept] == [0, 1, 2]


def test_a_single_shortlisted_instance_always_stays_however_poor():
    # No absolute threshold: the only Instance is the best of its shortlist.
    assert _kept([_shortlisted(0, -0.5, -0.75)]) == [0]


def test_an_empty_shortlist_stays_empty():
    assert _kept([]) == []

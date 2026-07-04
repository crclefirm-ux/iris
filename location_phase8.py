"""
Location service for Phase 8 / M6 (§6.5 location inference).

Location is inferred with a priority hierarchy — NO hardware GPS:

    Source 1 (highest): OCR from LLaVA — reads signage / storefront / menu
                        text in the video frames. A 'Yaffa Coffee House'
                        sign  →  location = 'Yaffa Coffee House'.
    Source 2:           Wi-Fi network name (SSID) — often carries the venue
                        or home identifier ('Yaffa_Guest', 'HomeNet').
    Source 3 (fallback): ip-api.com — free, no key, ~45 req/min, gives
                        city + region + approximate coords.

    Priority: OCR venue name  >  Wi-Fi SSID  >  IP geolocation city.

resolve_location() runs that whole chain and returns ONE location dict whose
`source` is one of the blueprint tokens:  ocr | wifi | ip | manual  (matching
the ChromaDB segment schema, §8.1). Crucially, it ALWAYS fills in `lat`/`lon`
(from the IP layer when the primary source has no coordinates of its own) so
the existing Location-tab map code — which reads loc["lat"] / loc["lon"]
directly — keeps working no matter which source won.

Location dict shape:
    {
        "location":     "Yaffa Coffee House", # display name (venue or city)
        "source":       "ocr",                # ocr | wifi | ip | manual
        "confidence":   "high",               # high | medium | low
        "label":        "Yaffa Coffee House", # OCR venue text, if any
        "ssid":         "Yaffa_Guest",        # Wi-Fi SSID, if any
        "lat":          39.7684,
        "lon":          -86.1581,
        "accuracy_m":   50000,
        "city":         "Indianapolis",
        "region":       "Indiana",
        "country":      "US",
        "provider":     "ip-api",
        "timestamp_iso":"2026-05-12T21:17:23"
    }

Stores location data as a sidecar JSON next to each WAV / clip:
    recording_..._chunk01.wav  →  recording_..._chunk01.location.json
The sidecar pattern keeps the media format clean and lets us add or remove
location data without rewriting recordings.
"""

import json
import os
import re
import time
import shutil
import subprocess
import threading
from datetime import datetime
from typing import Optional, Sequence

import requests


# ─────────────────────────────────────────────────────────────────────────
# Source 3 — IP geolocation (ip-api.com). Unchanged behaviour from Phase 8,
# except the `source` token is now the blueprint's "ip" (was "ip-api"); the
# original provider string is preserved under `provider`.
# ─────────────────────────────────────────────────────────────────────────
class LocationService:
    """Fetches and caches the host's IP-level geolocation."""

    def __init__(self, timeout_s: float = 5.0):
        self._timeout = timeout_s
        self._cached: Optional[dict] = None
        self._cached_at: float = 0.0
        self._cache_ttl: float = 3600.0    # 1 hour
        self._lock = threading.Lock()

    def get(self, force_refresh: bool = False) -> Optional[dict]:
        """Returns an IP-level location dict, or None if lookup failed."""
        with self._lock:
            now = time.time()
            if (not force_refresh
                    and self._cached is not None
                    and now - self._cached_at < self._cache_ttl):
                return self._cached

            loc = self._fetch_ip_api()
            if loc is not None:
                self._cached = loc
                self._cached_at = now
            return loc

    def _fetch_ip_api(self) -> Optional[dict]:
        try:
            resp = requests.get(
                "http://ip-api.com/json/",
                timeout=self._timeout,
                params={"fields": "status,country,regionName,city,lat,lon,query"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[location] ip-api fetch failed: {e}")
            return None

        if data.get("status") != "success":
            return None

        city = data.get("city", "")
        return {
            "location": city or data.get("regionName", "") or "Unknown",
            "source": "ip",              # blueprint token (ocr | wifi | ip | manual)
            "provider": "ip-api",
            "confidence": "low",         # city-level only
            "label": "",
            "ssid": "",
            "lat": float(data["lat"]),
            "lon": float(data["lon"]),
            "accuracy_m": 50000,         # ip-api is city-level; ~50 km guess
            "city": city,
            "region": data.get("regionName", ""),
            "country": data.get("country", ""),
            "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        }


# Shared IP service (so resolve_location() reuses one cache).
_ip_service: Optional[LocationService] = None
_ip_service_lock = threading.Lock()


def _get_ip_service() -> LocationService:
    global _ip_service
    if _ip_service is None:
        with _ip_service_lock:
            if _ip_service is None:
                timeout = 5.0
                try:
                    import config_phase9 as _cfg          # type: ignore
                    timeout = float(getattr(_cfg, "LOCATION_TIMEOUT_S", 5.0))
                except Exception:
                    pass
                _ip_service = LocationService(timeout_s=timeout)
    return _ip_service


# ─────────────────────────────────────────────────────────────────────────
# Source 2 — Wi-Fi SSID. Reads the currently-connected network name. This is
# the *host's* connected SSID (netsh / nmcli / airport), NOT the ESP32 audio
# link in wifi_reader_phase6.py — different concern, different reader.
# ─────────────────────────────────────────────────────────────────────────
def read_wifi_ssid(timeout_s: float = 3.0) -> str:
    """Return the SSID of the network this host is connected to, or ''.
    Cross-platform best effort; never raises."""
    try:
        # Windows: netsh wlan show interfaces
        if os.name == "nt":
            out = _run(["netsh", "wlan", "show", "interfaces"], timeout_s)
            if out:
                for line in out.splitlines():
                    # "    SSID                   : MyNetwork"
                    m = re.match(r"\s*SSID\s*:\s*(.+?)\s*$", line)
                    if m and "BSSID" not in line:
                        return m.group(1).strip()
            return ""
        # macOS: airport -I
        airport = ("/System/Library/PrivateFrameworks/Apple80211.framework/"
                   "Versions/Current/Resources/airport")
        if os.path.exists(airport):
            out = _run([airport, "-I"], timeout_s)
            if out:
                for line in out.splitlines():
                    m = re.match(r"\s*SSID:\s*(.+?)\s*$", line)
                    if m:
                        return m.group(1).strip()
        # Linux: nmcli / iwgetid
        if shutil.which("nmcli"):
            out = _run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                       timeout_s)
            if out:
                for line in out.splitlines():
                    if line.startswith("yes:"):
                        return line.split(":", 1)[1].strip()
        if shutil.which("iwgetid"):
            out = _run(["iwgetid", "-r"], timeout_s)
            if out:
                return out.strip().splitlines()[0].strip() if out.strip() else ""
    except Exception as e:
        print(f"[location] wifi ssid read failed: {e}")
    return ""


def _run(cmd, timeout_s: float) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout_s)
        return res.stdout or ""
    except Exception:
        return ""


def _ssid_to_location(ssid: str) -> str:
    """Map an SSID to a human venue/home name. Uses the optional
    SSID_LOCATION_MAP in config_phase9; otherwise cleans the raw SSID into a
    readable label ('Yaffa_Guest' → 'Yaffa Guest')."""
    if not ssid:
        return ""
    try:
        import config_phase9 as _cfg               # type: ignore
        mapping = getattr(_cfg, "SSID_LOCATION_MAP", None)
        if isinstance(mapping, dict):
            # case-insensitive lookup
            for k, v in mapping.items():
                if str(k).strip().lower() == ssid.strip().lower() and v:
                    return str(v)
    except Exception:
        pass
    cleaned = re.sub(r"[_\-]+", " ", ssid).strip()
    # Drop noise-y trailing tokens like guest / 5g / wifi / net.
    cleaned = re.sub(r"\b(guest|5g|2g|wifi|net|network)\b", "", cleaned,
                     flags=re.IGNORECASE).strip()
    return cleaned or ssid


# ─────────────────────────────────────────────────────────────────────────
# Source 1 — OCR via LLaVA. The fusion object owns the LLaVA client; we ask
# it to read signage from a few frames and hand back a venue name.
# ─────────────────────────────────────────────────────────────────────────
def _ocr_venue_from_frames(frames: Sequence, fusion) -> str:
    """Return a venue/place name read from signage in the frames, or ''.
    Delegates to fusion.read_signage() (iris_fusion) so the same warmed-up
    LLaVA client is reused. Safe if fusion/LLaVA is unavailable."""
    if not frames or fusion is None:
        return ""
    if not hasattr(fusion, "read_signage"):
        return ""
    try:
        venue = fusion.read_signage(list(frames))
    except Exception as e:
        print(f"[location] OCR read_signage failed: {e}")
        return ""
    venue = (venue or "").strip()
    # LLaVA is told to answer NONE when there is no readable venue text.
    if not venue or venue.strip(" .\"'").upper() in {"NONE", "N/A", "UNKNOWN"}:
        return ""
    return venue


# ─────────────────────────────────────────────────────────────────────────
# The fallback chain — OCR  →  Wi-Fi SSID  →  ip-api.
# ─────────────────────────────────────────────────────────────────────────
def resolve_location(frames: Optional[Sequence] = None,
                     fusion=None,
                     *,
                     use_ocr: bool = True,
                     use_wifi: bool = True,
                     use_ip: bool = True,
                     manual: Optional[str] = None) -> Optional[dict]:
    """Run the priority chain and return a single location dict, or None if
    every source failed. `lat`/`lon` are always present when at least the IP
    layer succeeds (so the map never sees a missing key)."""
    # Config toggles (blueprint allows disabling any layer).
    try:
        import config_phase9 as _cfg               # type: ignore
        use_ocr = use_ocr and bool(getattr(_cfg, "LOCATION_OCR_ENABLED", True))
        use_wifi = use_wifi and bool(getattr(_cfg, "LOCATION_WIFI_ENABLED", True))
        use_ip = use_ip and bool(getattr(_cfg, "LOCATION_IP_ENABLED", True))
    except Exception:
        pass

    # IP layer first (cheap, cached) — its coords back-fill lat/lon for every
    # source, but it is only the *primary* result if nothing better wins.
    ip_loc = _get_ip_service().get() if use_ip else None

    def _finish(primary: dict) -> dict:
        """Merge coords/city from the IP layer into a higher-priority
        primary result so lat/lon are always populated."""
        out = dict(primary)
        if ip_loc:
            for k in ("lat", "lon", "accuracy_m", "city", "region",
                      "country", "provider"):
                out.setdefault(k, ip_loc.get(k))
        # Guarantee the keys the map reads exist.
        out.setdefault("lat", _fallback_lat())
        out.setdefault("lon", _fallback_lon())
        out.setdefault("timestamp_iso",
                       datetime.now().isoformat(timespec="seconds"))
        return out

    # 0) Manual override (highest of all — user typed a correction).
    if manual and manual.strip():
        return _finish({
            "location": manual.strip(), "source": "manual",
            "confidence": "high", "label": manual.strip(), "ssid": "",
        })

    # 1) OCR venue name.
    if use_ocr:
        venue = _ocr_venue_from_frames(frames, fusion)
        if venue:
            print(f"[location] OCR → '{venue}'")
            return _finish({
                "location": venue, "source": "ocr", "confidence": "high",
                "label": venue, "ssid": "",
            })

    # 2) Wi-Fi SSID.
    if use_wifi:
        ssid = read_wifi_ssid()
        if ssid:
            name = _ssid_to_location(ssid)
            print(f"[location] Wi-Fi SSID '{ssid}' → '{name}'")
            return _finish({
                "location": name or ssid, "source": "wifi",
                "confidence": "medium", "label": "", "ssid": ssid,
            })

    # 3) IP fallback.
    if ip_loc:
        return ip_loc

    return None


def _fallback_lat() -> float:
    try:
        import config_phase9 as _cfg               # type: ignore
        return float(getattr(_cfg, "MAP_FALLBACK_LAT", 39.7684))
    except Exception:
        return 39.7684


def _fallback_lon() -> float:
    try:
        import config_phase9 as _cfg               # type: ignore
        return float(getattr(_cfg, "MAP_FALLBACK_LON", -86.1581))
    except Exception:
        return -86.1581


# ─────────────────────────────────────────────────────────────────────────
# Sidecar read/write — unchanged from Phase 8.
# ─────────────────────────────────────────────────────────────────────────
def save_location_sidecar(wav_path: str, location: dict) -> None:
    """Write a .location.json next to the given WAV / clip."""
    if location is None:
        return
    sidecar = os.path.splitext(wav_path)[0] + ".location.json"
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(location, f, indent=2)
    except Exception as e:
        print(f"[location] could not save sidecar: {e}")


def load_location_sidecar(wav_path: str) -> Optional[dict]:
    """Read the .location.json next to the given WAV / clip, if present."""
    sidecar = os.path.splitext(wav_path)[0] + ".location.json"
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
import argparse
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Kurs tetap Transfermarkt (1 EUR = Rp 17.381,8181818...)
TM_EUR_TO_IDR = 191200.0 / 11.0

# Pemetaan Position ID Transfermarkt -> Nama Posisi Bahasa Indonesia (kompatibel penuh dengan dataset)
POSITION_ID_MAP = {
    1: ("Kiper", "Goalkeeper"),
    2: ("Bek", "Defender"),
    3: ("Bek-Tengah", "Defender"),
    4: ("Bek-Kiri", "Defender"),
    5: ("Bek-Kanan", "Defender"),
    6: ("Gel. Bertahan", "Midfielder"),
    7: ("Gel. Tengah", "Midfielder"),
    8: ("Gelandang Kanan", "Midfielder"),
    9: ("Gelandang Kiri", "Midfielder"),
    10: ("Gel. Serang", "Midfielder"),
    11: ("Sayap Kiri", "Attack"),
    12: ("Sayap Kanan", "Attack"),
    13: ("Depan-Kedua", "Attack"),
    14: ("Depan-Tengah", "Attack"),
}

# Pemetaan Country ID Transfermarkt -> Nama Negara Bahasa Indonesia
COUNTRY_ID_MAP = {
    1: "Afganistan",
    2: "Mesir",
    3: "Albania",
    4: "Aljazair",
    6: "Angola",
    8: "Guinea Khatulistiwa",
    9: "Argentina",
    10: "Armenia",
    12: "Australia",
    13: "Azerbaijan",
    15: "Bahrain",
    18: "Belarus",
    19: "Belgia",
    21: "Benin",
    23: "Bolivia",
    24: "Bosnia-Herzegovina",
    25: "Bosnia-Herzegovina",
    26: "Brasil",
    27: "Brunei Darussalam",
    28: "Bulgaria",
    29: "Brunei Darussalam",
    30: "Burundi",
    31: "Kamerun",
    32: "Tanjung Verde",
    33: "Kolombia",
    34: "Tiongkok",
    35: "Cile",
    36: "Kosta Rika",
    37: "Kroasia",
    38: "Pantai Gading",
    39: "Denmark",
    40: "Jerman",
    44: "Ekuador",
    49: "Finlandia",
    50: "Prancis",
    52: "Georgia",
    54: "Ghana",
    56: "Yunani",
    58: "Guatemala",
    59: "Guinea",
    60: "Guinea-Bissau",
    62: "Haiti",
    64: "Honduras",
    65: "Hong Kong",
    66: "India",
    68: "Indonesia",
    70: "Irak",
    71: "Iran",
    72: "Irlandia",
    73: "Islandia",
    74: "Israel",
    75: "Italia",
    76: "Jamaika",
    77: "Jepang",
    78: "Yordania",
    79: "Kamboja",
    80: "Kanada",
    81: "Kazakhstan",
    83: "Kolombia",
    85: "Kongo",
    86: "Kirgizstan",
    87: "Korea Selatan",
    90: "Kirgizstan",
    91: "Laos",
    92: "Lebanon",
    95: "Liberia",
    98: "Luksemburg",
    99: "Slovenia",
    100: "Makedonia Utara",
    101: "Madagaskar",
    103: "Malaysia",
    104: "Mali",
    105: "Malta",
    106: "Malta",
    107: "Maroko",
    109: "Maroko",
    110: "Meksiko",
    114: "Norwegia",
    116: "Myanmar",
    117: "Namibia",
    120: "Selandia Baru",
    122: "Belanda",
    123: "Selandia Baru",
    124: "Nigeria",
    125: "Nigeria",
    126: "Norwegia",
    127: "Austria",
    131: "Palestina",
    132: "Paraguay",
    133: "Peru",
    134: "Filipina",
    135: "Polandia",
    136: "Portugal",
    140: "Rumania",
    141: "Rusia",
    145: "Sao Tome dan Principe",
    147: "Swedia",
    148: "Swiss",
    149: "Senegal",
    153: "Singapura",
    154: "Slowakia",
    155: "Slovenia",
    157: "Spanyol",
    159: "Afrika Selatan",
    161: "Suriname",
    162: "Suriah",
    163: "Tajikistan",
    164: "Tionghoa Taipei",
    165: "Tajikistan",
    167: "Thailand",
    168: "Togo",
    170: "Trinidad dan Tobago",
    171: "Turkmenistan",
    172: "Turki",
    173: "Tunisia",
    176: "Uganda",
    177: "Ukraina",
    178: "Hungaria",
    179: "Uruguay",
    180: "Uzbekistan",
    182: "Venezuela",
    183: "Uzbekistan",
    184: "Amerika Serikat",
    185: "Vietnam",
    188: "Siprus",
    189: "Inggris",
    190: "Skotlandia",
    191: "Wales",
    215: "Serbia",
    216: "Montenegro",
    232: "Curacao",
    241: "Timor-Leste",
    242: "Timor-Leste",
    244: "Kosovo",
    260: "Curacao",
    262: "Sudan Selatan",
    269: "Bonaire",
}

ASEAN_COUNTRIES = {
    "Indonesia", "Thailand", "Malaysia", "Vietnam", "Singapura",
    "Filipina", "Myanmar", "Kamboja", "Laos", "Brunei Darussalam", "Timor-Leste",
}

AFC_COUNTRIES = ASEAN_COUNTRIES | {
    "Afganistan", "Jepang", "Korea Selatan", "Australia", "Iran", "Uzbekistan", "Arab Saudi",
    "Qatar", "Uni Emirat Arab", "Irak", "Yordania", "Palestina", "Suriah",
    "Lebanon", "Bahrain", "Oman", "Kuwait", "Kirgizstan", "Tajikistan",
    "Turkmenistan", "India", "Tiongkok", "Hong Kong", "Tionghoa Taipei", "Korea Utara",
}

FOOT_MAP = {
    "right": "kanan",
    "left": "kiri",
    "both": "keduanya",
    "kanan": "kanan",
    "kiri": "kiri",
    "keduanya": "keduanya",
}


class RealScout:
    def __init__(self, include_stats=True, mode="auto"):
        """
        Inisialisasi RealScout v6.0
        :param include_stats: Ambil juga statistik penampilan/gol/menit bermain
        :param mode: 'auto' (HTML dengan fallback API), 'html' (khusus HTML), atau 'api' (khusus TMAPI cepat)
        """
        self.raw_path = "data/raw"
        self.processed_path = "data/processed"
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)

        self.include_stats = include_stats
        self.mode = mode

        # Base URL Transfermarkt (Versi Indonesia) & Official JSON API
        self.base_url = "https://www.transfermarkt.co.id"
        self.api_base_url = "https://tmapi-alpha.transfermarkt.technology"

        # Target Liga: Ekspansi Liga 1, Liga 2, Liga 3 Indonesia + 3 Liga Top ASEAN
        self.targets = {
            "Indonesia (Liga 1)": "https://www.transfermarkt.co.id/super-league/startseite/wettbewerb/IN1L",
            "Indonesia (Liga 2)": "https://www.transfermarkt.co.id/championship-indonesia/startseite/wettbewerb/ILI2",
            "Indonesia (Liga 3)": "https://www.transfermarkt.co.id/liga-nusantara/startseite/wettbewerb/IN3L",
            "Thailand": "https://www.transfermarkt.co.id/thai-league/startseite/wettbewerb/THA1",
            "Malaysia": "https://www.transfermarkt.co.id/super-league/startseite/wettbewerb/MYS1",
            "Vietnam": "https://www.transfermarkt.co.id/v-league-1/startseite/wettbewerb/VIE1",
        }

        self.competition_meta = {
            "Indonesia (Liga 1)": {"comp_id": "IN1L", "country": "Indonesia", "tier": 1},
            "Indonesia (Liga 2)": {"comp_id": "ILI2", "country": "Indonesia", "tier": 2},
            "Indonesia (Liga 3)": {"comp_id": "IN3L", "country": "Indonesia", "tier": 3},
            "Thailand": {"comp_id": "THA1", "country": "Thailand", "tier": 1},
            "Malaysia": {"comp_id": "MYS1", "country": "Malaysia", "tier": 1},
            "Vietnam": {"comp_id": "VIE1", "country": "Vietnam", "tier": 1},
        }

    # =========================================================================
    # PARSING & FORMATTING HELPERS (BUG FIXES)
    # =========================================================================
    @staticmethod
    def parse_market_value(market_value_str):
        """
        Memperbaiki bug konversi Market Value 10x lipat.
        Contoh:
          "Rp4,78Mlyr." -> 4_780_000_000
          "Rp434,54Jt." -> 434_540_000
          "Rp86,91Jt."  -> 86_910_000
          "-" / "Rp0"   -> 0
        """
        if not market_value_str or not isinstance(market_value_str, str):
            return 0

        s = market_value_str.strip()
        if s in ("-", "Rp0", "0", ""):
            return 0

        multiplier = 1
        s_lower = s.lower()
        if "mlyr" in s_lower or "miliar" in s_lower:
            multiplier = 1_000_000_000
        elif "jt" in s_lower or "juta" in s_lower:
            multiplier = 1_000_000
        elif "rb" in s_lower or "ribu" in s_lower:
            multiplier = 1_000

        # Ambil bagian angka (termasuk koma desimal)
        num_match = re.search(r"(\d+(?:[.,]\d+)?)", s)
        if not num_match:
            return 0

        num_str = num_match.group(1).replace(".", "").replace(",", ".")
        try:
            return int(round(float(num_str) * multiplier))
        except ValueError:
            return 0

    @staticmethod
    def format_market_value_idr(val_idr):
        """Format nilai Rupiah integer menjadi format standar Transfermarkt Indonesia."""
        if not val_idr or val_idr <= 0:
            return "-"
        if val_idr >= 1_000_000_000:
            val_m = val_idr / 1_000_000_000.0
            formatted = f"{val_m:.2f}".replace(".", ",")
            return f"Rp{formatted}Mlyr."
        elif val_idr >= 1_000_000:
            val_jt = val_idr / 1_000_000.0
            formatted = f"{val_jt:.2f}".replace(".", ",")
            return f"Rp{formatted}Jt."
        else:
            return f"Rp{val_idr:,}".replace(",", ".")

    @staticmethod
    def parse_birth_date_and_age(text):
        """
        Memperbaiki bug ekstraksi usia (age=0 / tertukar nomor punggung).
        Contoh input: "27 Feb 2002 (24)" -> ("27 Feb 2002", 24)
        """
        if not text or not isinstance(text, str):
            return "-", 0
        cleaned = text.strip()
        # Cari angka di dalam tanda kurung: (24)
        age_match = re.search(r"\((\d{1,2})\)", cleaned)
        if age_match:
            age = int(age_match.group(1))
            birth_date = re.sub(r"\s*\(\d{1,2}\)", "", cleaned).strip() or "-"
            return birth_date, age

        # Jika hanya angka murni (misal di tabel leistungsdaten)
        if cleaned.isdigit():
            val = int(cleaned)
            if 14 <= val <= 50:
                return "-", val

        return cleaned if cleaned else "-", 0

    @staticmethod
    def parse_height_cm(text):
        """
        Konversi string tinggi badan (contoh: '1,79m' atau '1.79 m' atau 1.79) menjadi cm (179).
        """
        if text is None:
            return 0
        if isinstance(text, (int, float)):
            if 1.3 <= float(text) <= 2.3:
                return int(round(float(text) * 100))
            if 130 <= float(text) <= 230:
                return int(round(float(text)))
            return 0

        s = str(text).strip().lower().replace("m", "").strip()
        if not s or s == "-":
            return 0

        # Contoh: "1,79" -> 179
        m = re.search(r"(\d)[,.](\d{2})", s)
        if m:
            return int(m.group(1)) * 100 + int(m.group(2))

        m_digits = re.search(r"(\d{3})", s)
        if m_digits:
            return int(m_digits.group(1))
        return 0

    @staticmethod
    def classify_quota_status(nationality, league_label):
        """
        Klasifikasi status kuota pemain berdasarkan kewarganegaraan:
        - Lokal (WNI untuk Liga Indonesia, atau warga negara domestik liga terkait)
        - ASEAN
        - Asia (AFC)
        - Asing Non-Asia
        """
        if not nationality or nationality == "Unknown":
            return "Lokal"

        nat_clean = nationality.strip()
        if "Indonesia" in league_label and nat_clean == "Indonesia":
            return "Lokal"
        if league_label == "Thailand" and nat_clean == "Thailand":
            return "Lokal"
        if league_label == "Malaysia" and nat_clean == "Malaysia":
            return "Lokal"
        if league_label == "Vietnam" and nat_clean == "Vietnam":
            return "Lokal"

        if nat_clean in ASEAN_COUNTRIES:
            return "ASEAN"
        if nat_clean in AFC_COUNTRIES:
            return "Asia (AFC)"
        return "Asing Non-Asia"

    @staticmethod
    def infer_position_group(position_name):
        """Menentukan grup posisi utama dari nama posisi Bahasa Indonesia."""
        p = str(position_name).lower()
        if "kiper" in p or "goalkeeper" in p:
            return "Goalkeeper"
        if "bek" in p or "back" in p or "defender" in p:
            return "Defender"
        if "gel" in p or "midfield" in p:
            return "Midfielder"
        return "Attack"

    # =========================================================================
    # HTTP & HTML SCRAPING ENGINE
    # =========================================================================
    def _get_headers(self):
        """Memalsukan identitas browser agar tidak ditolak server."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        ]
        return {
            "User-Agent": random.choice(user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        }

    def get_soup(self, url, min_delay=2.0, max_delay=4.5):
        """Helper function untuk request + parsing dengan error handling."""
        try:
            delay = random.uniform(min_delay, max_delay)
            print(f"⏳ Waiting {delay:.1f}s before fetching {url.split('/')[-1]}...")
            time.sleep(delay)

            response = requests.get(url, headers=self._get_headers(), timeout=15)
            if response.status_code == 200:
                return BeautifulSoup(response.content, "html.parser")
            elif response.status_code in (403, 405):
                print(f"⚠️ WAF/Akses Ditolak ({response.status_code}) pada {url}.")
                return None
            else:
                print(f"⚠️ HTTP Error {response.status_code} ke {url}")
                return None
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return None

    @staticmethod
    def _to_detailed_kader_url(team_url):
        """
        Mengubah URL tim standar (/startseite/verein/...) menjadi URL Detailed Squad (/kader/verein/.../plus/1).
        """
        url = team_url.rstrip("/")
        if "/startseite/verein/" in url:
            url = url.replace("/startseite/verein/", "/kader/verein/")
        if not url.endswith("/plus/1"):
            url = url + "/plus/1"
        return url

    @staticmethod
    def _to_leistungsdaten_url(team_url):
        """
        Mengubah URL tim menjadi URL Statistik Performa (/leistungsdaten/verein/.../plus/1).
        """
        url = team_url.rstrip("/")
        url = re.sub(r"/saison_id/\d+", "", url)
        url = re.sub(r"/plus/1$", "", url)
        if "/startseite/verein/" in url:
            url = url.replace("/startseite/verein/", "/leistungsdaten/verein/")
        elif "/kader/verein/" in url:
            url = url.replace("/kader/verein/", "/leistungsdaten/verein/")
        return url + "/plus/1"

    def scrape_teams_from_league(self, league_url):
        """Mengambil URL setiap tim dari halaman klasemen/daftar klub liga."""
        soup = self.get_soup(league_url)
        if not soup:
            return []

        team_urls = []
        tables = soup.find_all("table", class_="items")
        for table in tables:
            links = table.find_all("td", class_="hauptlink")
            for link in links:
                a_tag = link.find("a", href=True)
                if a_tag:
                    href = a_tag["href"]
                    if "/verein/" in href and ("/startseite/" in href or "/kader/" in href):
                        full_url = self.base_url + href
                        # Normalisasi berdasarkan verein ID agar tidak duplikat
                        if full_url not in team_urls:
                            team_urls.append(full_url)

        # Deduplikasi berdasarkan club/verein ID
        unique_by_id = {}
        for u in team_urls:
            m = re.search(r"/verein/(\d+)", u)
            cid = m.group(1) if m else u
            if cid not in unique_by_id:
                unique_by_id[cid] = u
        return list(unique_by_id.values())

    def parse_leistungsdaten_soup(self, soup):
        """
        Mengekstrak statistik performa pemain dari halaman /leistungsdaten/verein/<id>/plus/1.
        Mengembalikan dict: {player_id_or_name: {appearances, goals, assists, yellow_cards, red_cards, ppg, minutes_played}}
        """
        stats_map = {}
        if not soup:
            return stats_map

        table = soup.find("table", class_="items")
        if not table:
            return stats_map

        rows = table.find_all("tr", class_=["odd", "even"])
        for row in rows:
            try:
                cells = row.find_all("td", recursive=False)
                if len(cells) < 15:
                    continue

                # Cari player_id atau player_name
                player_id = ""
                a_player = cells[1].find("a", href=re.compile(r"/profil/spieler/\d+"))
                if not a_player:
                    a_player = cells[1].find("a", href=True)
                player_name = a_player.text.strip() if a_player else ""
                if a_player and "href" in a_player.attrs:
                    m_id = re.search(r"/spieler/(\d+)", a_player["href"])
                    if m_id:
                        player_id = m_id.group(1)

                def _to_int(val_str):
                    v = val_str.strip()
                    return int(v) if v.isdigit() else 0

                in_squad = _to_int(cells[4].text)
                appearances = _to_int(cells[5].text)
                goals = _to_int(cells[6].text)
                assists = _to_int(cells[7].text)
                yellow_cards = _to_int(cells[8].text)
                second_yellow = _to_int(cells[9].text)
                red_cards = _to_int(cells[10].text) + second_yellow

                ppg_raw = cells[13].text.strip().replace(",", ".")
                try:
                    ppg = float(ppg_raw) if ppg_raw not in ("-", "") else 0.0
                except ValueError:
                    ppg = 0.0

                min_raw = cells[14].text.strip().replace("'", "").replace(".", "")
                minutes_played = int(min_raw) if min_raw.isdigit() else 0

                stat_obj = {
                    "in_squad": in_squad,
                    "appearances": appearances,
                    "goals": goals,
                    "assists": assists,
                    "yellow_cards": yellow_cards,
                    "red_cards": red_cards,
                    "ppg": ppg,
                    "minutes_played": minutes_played,
                }
                if player_id:
                    stats_map[player_id] = stat_obj
                if player_name:
                    stats_map[player_name] = stat_obj
            except Exception:
                continue

        return stats_map

    def parse_squad_soup(self, soup, league_label, stats_map=None):
        """
        Mem-parsing halaman skuad tim (/kader/verein/.../plus/1 maupun /startseite/verein/...).
        """
        if not soup:
            return []

        stats_map = stats_map or {}
        meta = self.competition_meta.get(
            league_label, {"comp_id": "UNK", "country": league_label, "tier": 1}
        )

        try:
            team_name = soup.find("h1", class_="data-header__headline-wrapper").text.strip()
        except Exception:
            team_name = "Unknown Team"

        table = soup.find("table", class_="items")
        if not table:
            return []

        players_data = []
        rows = table.find_all("tr", class_=["odd", "even"])

        for row in rows:
            try:
                cells = row.find_all("td", recursive=False)
                if len(cells) < 5:
                    continue

                # 0. Nomor Punggung
                shirt_raw = cells[0].text.strip()
                shirt_number = int(shirt_raw) if shirt_raw.isdigit() else 0

                # 1. Nama, ID & Posisi
                td_name = row.find("table", class_="inline-table")
                if not td_name:
                    continue

                hauptlink_td = td_name.find("td", class_="hauptlink")
                name_tag = hauptlink_td.find("a") if hauptlink_td else None
                if not name_tag:
                    continue

                player_name = name_tag.text.strip()
                player_id = ""
                if name_tag.has_attr("href"):
                    m_id = re.search(r"/spieler/(\d+)", name_tag["href"])
                    if m_id:
                        player_id = m_id.group(1)

                inline_trs = td_name.find_all("tr")
                pos_tag = inline_trs[-1].find("td") if len(inline_trs) > 1 else None
                position = pos_tag.text.strip() if pos_tag else "Unknown"
                position_group = self.infer_position_group(position)

                # 2. Tanggal Lahir & Usia (td[2]) - FIX BUG USIA!
                birth_date, age = self.parse_birth_date_and_age(cells[2].text)

                # 3. Kewarganegaraan (td[3])
                nat_imgs = [
                    (img.get("title") or img.get("alt")).strip()
                    for img in cells[3].find_all("img")
                    if img.get("title") or img.get("alt")
                ]
                nationality = nat_imgs[0] if nat_imgs else meta["country"]
                quota_status = self.classify_quota_status(nationality, league_label)

                # Cek apakah mode Detailed (/plus/1 -> 10 kolom) atau Standard (5 kolom)
                if len(cells) >= 10:
                    height_cm = self.parse_height_cm(cells[4].text)
                    foot_raw = cells[5].text.strip().lower()
                    foot = FOOT_MAP.get(foot_raw, foot_raw if foot_raw and foot_raw != "-" else "kanan")
                    joined_date = cells[6].text.strip() or "-"
                    prev_imgs = [
                        (img.get("title") or img.get("alt")).strip()
                        for img in cells[7].find_all("img")
                        if img.get("title") or img.get("alt")
                    ]
                    previous_club = prev_imgs[0] if prev_imgs else "-"
                    contract_expiry = cells[8].text.strip() or "-"
                    mv_cell = cells[9]
                else:
                    height_cm = 0
                    foot = "kanan"
                    joined_date = "-"
                    previous_club = "-"
                    contract_expiry = "-"
                    mv_cell = row.find("td", class_="rechts hauptlink") or cells[-1]

                market_value_raw = mv_cell.text.strip() if mv_cell else "-"
                if not market_value_raw:
                    market_value_raw = "-"
                market_value_est = self.parse_market_value(market_value_raw)

                # Gabungkan dengan statistik performa (jika ada)
                p_stats = stats_map.get(player_id) or stats_map.get(player_name) or {}

                players_data.append({
                    "player_id": player_id,
                    "shirt_number": shirt_number,
                    "player_name": player_name,
                    "team": team_name,
                    "league_country": league_label,
                    "competition_id": meta["comp_id"],
                    "league_tier": meta["tier"],
                    "position": position,
                    "position_group": position_group,
                    "age": int(age),
                    "birth_date": birth_date,
                    "is_u22": bool(0 < int(age) <= 22),
                    "nationality": nationality,
                    "quota_status": quota_status,
                    "height_cm": int(height_cm),
                    "foot": foot,
                    "joined_date": joined_date,
                    "previous_club": previous_club,
                    "contract_expiry": contract_expiry,
                    "market_value_raw": market_value_raw,
                    "market_value_est": int(market_value_est),
                    "prev_market_value_est": int(market_value_est),
                    "mv_trend": "STABLE",
                    "appearances": int(p_stats.get("appearances", 0)),
                    "goals": int(p_stats.get("goals", 0)),
                    "assists": int(p_stats.get("assists", 0)),
                    "minutes_played": int(p_stats.get("minutes_played", 0)),
                    "ppg": float(p_stats.get("ppg", 0.0)),
                    "yellow_cards": int(p_stats.get("yellow_cards", 0)),
                    "red_cards": int(p_stats.get("red_cards", 0)),
                    "scraped_date": datetime.now().strftime("%Y-%m-%d"),
                })
            except Exception:
                continue

        return players_data

    def scrape_players_from_team(self, team_url, country_name):
        """Mengambil data skuad lengkap (+ statistik opsional) dari halaman tim via HTML."""
        kader_url = self._to_detailed_kader_url(team_url)
        soup = self.get_soup(kader_url)
        if not soup:
            return []

        stats_map = {}
        if self.include_stats:
            stats_url = self._to_leistungsdaten_url(team_url)
            stats_soup = self.get_soup(stats_url, min_delay=1.5, max_delay=3.0)
            stats_map = self.parse_leistungsdaten_soup(stats_soup)

        return self.parse_squad_soup(soup, country_name, stats_map=stats_map)

    # =========================================================================
    # TMAPI OFFICIAL JSON ENGINE (FALLBACK ANTI-WAF 403/405 & FAST BATCH MODE)
    # =========================================================================
    def _api_get(self, endpoint, params=None, retries=3):
        url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
        for attempt in range(retries):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                    timeout=15,
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
            time.sleep(0.8 * (attempt + 1))
        return None

    def scrape_league_via_api(self, league_label, max_teams=None):
        """
        Mengambil seluruh klub dan pemain di satu liga melalui Official JSON API Transfermarkt
        (tmapi-alpha.transfermarkt.technology). Sangat cepat dan bebas blokir WAF CloudFront.
        """
        meta = self.competition_meta[league_label]
        comp_id = meta["comp_id"]
        print(f"⚡ [TMAPI Engine] Mengambil data kompetisi {league_label} ({comp_id})...")

        table_json = self._api_get(f"competition/{comp_id}/table")
        if not table_json or not table_json.get("data"):
            print(f"❌ [TMAPI] Gagal mengambil daftar klub untuk {comp_id}.")
            return []

        tables = table_json["data"].get("tables", [])
        club_ids = []
        for t in tables:
            for c in t.get("clubs", []):
                cid = str(c.get("clubId", "")).strip()
                if cid and cid not in club_ids:
                    club_ids.append(cid)

        if max_teams:
            club_ids = club_ids[:max_teams]

        if not club_ids:
            return []

        # Batch fetch nama klub
        club_names = {}
        for i in range(0, len(club_ids), 25):
            batch_cids = club_ids[i : i + 25]
            q = "&".join(f"ids[]={cid}" for cid in batch_cids)
            clubs_json = self._api_get(f"clubs?{q}")
            if clubs_json and clubs_json.get("data"):
                for c_obj in clubs_json["data"]:
                    club_names[str(c_obj["id"])] = c_obj.get("name", f"Club {c_obj['id']}")

        print(f"📊 Menemukan {len(club_ids)} klub di {league_label}. Mengambil daftar skuad...")

        # Ambil daftar skuad per klub secara paralel ringan
        club_squads = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            future_to_cid = {
                pool.submit(self._api_get, f"club/{cid}/squad"): cid for cid in club_ids
            }
            for fut in as_completed(future_to_cid):
                cid = future_to_cid[fut]
                sq_json = fut.result()
                if sq_json and sq_json.get("data"):
                    club_squads[cid] = sq_json["data"].get("squad", [])

        # Kumpulkan semua playerId untuk di-fetch secara batch
        all_player_ids = []
        player_club_map = {}
        for cid, sq_list in club_squads.items():
            for item in sq_list:
                pid = str(item.get("playerId", "")).strip()
                if pid:
                    all_player_ids.append(pid)
                    player_club_map[pid] = {
                        "club_id": cid,
                        "team_name": club_names.get(cid, f"Club {cid}"),
                        "shirt_number": item.get("shirtNumber") or 0,
                    }

        unique_pids = list(dict.fromkeys(all_player_ids))
        print(f"   👥 Mengambil profil detail {len(unique_pids)} pemain di {league_label}...")

        player_profiles = {}
        batch_size = 30
        pid_batches = [
            unique_pids[i : i + batch_size] for i in range(0, len(unique_pids), batch_size)
        ]

        def _fetch_player_batch(batch):
            q = "&".join(f"ids[]={pid}" for pid in batch)
            return self._api_get(f"players?{q}")

        with ThreadPoolExecutor(max_workers=6) as pool:
            for res in pool.map(_fetch_player_batch, pid_batches):
                if res and res.get("data"):
                    for p_obj in res["data"]:
                        player_profiles[str(p_obj["id"])] = p_obj

        players_data = []
        today_str = datetime.now().strftime("%Y-%m-%d")

        for pid, club_info in player_club_map.items():
            p = player_profiles.get(pid)
            if not p:
                continue

            player_name = p.get("name") or p.get("shortName") or f"Player {pid}"
            life = p.get("lifeDates") or {}
            age = int(life.get("age") or 0)
            birth_date = life.get("dateOfBirth") or "-"

            attrs = p.get("attributes") or {}
            pos_id = attrs.get("positionId") or 0
            pos_name, pos_group = POSITION_ID_MAP.get(pos_id, ("Gel. Tengah", "Midfielder"))

            height_cm = self.parse_height_cm(attrs.get("height"))
            foot_obj = attrs.get("preferredFoot") or {}
            foot_raw = (foot_obj.get("name") or "").lower()
            foot = FOOT_MAP.get(foot_raw, "kanan")

            contract_until = attrs.get("contractUntil") or "-"
            if contract_until in ("0000-00-00", "None", None):
                contract_until = "-"

            # Kewarganegaraan
            nat_details = (p.get("nationalityDetails") or {}).get("nationalities") or {}
            nat_id = nat_details.get("nationalityId") or 0
            nationality = COUNTRY_ID_MAP.get(nat_id, meta["country"] if nat_id == 0 else f"Negara #{nat_id}").strip()
            quota_status = self.classify_quota_status(nationality, league_label)

            # Riwayat klub / tanggal bergabung dari clubAssignments
            joined_date = "-"
            for ca in p.get("clubAssignments") or []:
                if ca.get("type") == "current" and ca.get("start"):
                    joined_date = ca.get("start")
                    break

            former_note = attrs.get("formerClubsNote") or "-"

            # Market Value (Konversi EUR -> IDR menggunakan kurs Transfermarkt)
            mv_details = p.get("marketValueDetails") or {}
            curr_mv_eur = (mv_details.get("current") or {}).get("value") or 0
            prev_mv_eur = (mv_details.get("previous") or {}).get("value") or curr_mv_eur
            delta_type = (mv_details.get("delta") or {}).get("type") or "STABLE"

            market_value_est = int(round(curr_mv_eur * TM_EUR_TO_IDR)) if curr_mv_eur > 0 else 0
            prev_market_value_est = int(round(prev_mv_eur * TM_EUR_TO_IDR)) if prev_mv_eur > 0 else market_value_est
            market_value_raw = self.format_market_value_idr(market_value_est)

            players_data.append({
                "player_id": pid,
                "shirt_number": int(club_info["shirt_number"] or 0),
                "player_name": player_name,
                "team": club_info["team_name"],
                "league_country": league_label,
                "competition_id": comp_id,
                "league_tier": meta["tier"],
                "position": pos_name,
                "position_group": pos_group,
                "age": age,
                "birth_date": birth_date,
                "is_u22": bool(0 < age <= 22),
                "nationality": nationality,
                "quota_status": quota_status,
                "height_cm": height_cm,
                "foot": foot,
                "joined_date": joined_date,
                "previous_club": former_note[:60] if former_note else "-",
                "contract_expiry": contract_until,
                "market_value_raw": market_value_raw,
                "market_value_est": market_value_est,
                "prev_market_value_est": prev_market_value_est,
                "mv_trend": delta_type,
                "appearances": 0,
                "goals": 0,
                "assists": 0,
                "minutes_played": 0,
                "ppg": 0.0,
                "yellow_cards": 0,
                "red_cards": 0,
                "scraped_date": today_str,
            })

        return players_data

    # =========================================================================
    # HISTORICAL CSV REPAIR UTILITY
    # =========================================================================
    def repair_historical_csvs(self):
        """
        Memperbaiki bug market_value_est (meleset 10x lipat) pada seluruh file CSV historis di data/raw/.
        """
        raw_dir = Path(self.raw_path)
        repaired_count = 0
        for csv_file in sorted(raw_dir.glob("real_scout_*.csv")):
            try:
                df_hist = pd.read_csv(csv_file)
                if "market_value_raw" in df_hist.columns:
                    df_hist["market_value_est"] = df_hist["market_value_raw"].apply(
                        self.parse_market_value
                    )
                    df_hist.to_csv(csv_file, index=False)
                    repaired_count += 1
            except Exception as e:
                print(f"⚠️ Gagal memperbaiki {csv_file.name}: {e}")
        print(f"🛠️ Berhasil memperbaiki konversi market_value_est pada {repaired_count} file historis di {self.raw_path}/.")

    # =========================================================================
    # MAIN RUNNER
    # =========================================================================
    def run(self, max_teams_per_league=None):
        print("🚀 Memulai Garuda Scout AI v6.0 Data Ingestion Engine...")
        all_players = []

        for league_label, league_url in self.targets.items():
            print(f"\n🌍 Masuk ke Kompetisi: {league_label}")
            print(f"🔗 URL: {league_url}")

            league_players = []

            if self.mode in ("html", "auto"):
                team_links = self.scrape_teams_from_league(league_url)
                if team_links:
                    if max_teams_per_league:
                        team_links = team_links[:max_teams_per_league]
                    print(f"📊 [HTML Engine] Menemukan {len(team_links)} tim di {league_label}.")
                    for team_url in team_links:
                        try:
                            squad = self.scrape_players_from_team(team_url, league_label)
                            league_players.extend(squad)
                        except Exception as e:
                            print(f"❌ Error scraping team {team_url}: {e}")
                elif self.mode == "auto":
                    print(f"🔄 [Auto-Fallback] HTML diblokir WAF, beralih ke TMAPI JSON Engine untuk {league_label}...")

            if not league_players and self.mode in ("api", "auto"):
                league_players = self.scrape_league_via_api(
                    league_label, max_teams=max_teams_per_league
                )

            print(f"   ✅ Terkumpul {len(league_players)} pemain dari {league_label}.")
            all_players.extend(league_players)

        if all_players:
            df = pd.DataFrame(all_players)
            df = df.drop_duplicates(subset=["player_name", "team"], keep="first")

            filename = f"{self.raw_path}/real_scout_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            df.to_csv(f"{self.processed_path}/master_player_db.csv", index=False)

            print("\n✅ SCRAPING SELESAI!")
            print(f"Total Pemain: {len(df)} dari {df['team'].nunique()} Klub ({df['league_country'].nunique()} Liga)")
            print(f"File Raw tersimpan di: {filename}")
            print(f"Database Utama diperbarui: {self.processed_path}/master_player_db.csv")
            return df
        else:
            print("\n❌ Gagal mendapatkan data.")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Garuda Scout AI v6.0 Scraper Engine")
    parser.add_argument(
        "--mode",
        choices=["auto", "html", "api"],
        default="auto",
        help="Mode scraping: 'auto' (HTML + fallback API), 'html', atau 'api'",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Lewati pengambilan halaman statistik performa (/leistungsdaten)",
    )
    parser.add_argument(
        "--max-teams",
        type=int,
        default=None,
        help="Batasi jumlah klub per liga (untuk testing cepat)",
    )
    parser.add_argument(
        "--repair-history",
        action="store_true",
        help="Perbaiki bug market_value_est 10x pada seluruh file CSV historis di data/raw/",
    )
    args = parser.parse_args()

    scout = RealScout(include_stats=not args.no_stats, mode=args.mode)
    if args.repair_history:
        scout.repair_historical_csvs()
    scout.run(max_teams_per_league=args.max_teams)

"""
codice_6_comparison.py
======================
Modulo comparativo per "La corsa all'energia: chi vince la transizione?"

Uso:
    1. Esegui codice_5_export.py con PAESE = "United States"  -> genera risultati_United_States.npz
    2. Esegui codice_5_export.py con PAESE = "China"          -> genera risultati_China.npz
    3. Esegui questo file                                      -> genera fig5_comparison.png

Struttura:
    EnergyRaceLoader   — carica e valida un .npz, espone attributi tipizzati
    EnergyRaceScorer   — calcola le 10 metriche normalizzate [0-100] per la scorecard
    EnergyRaceComparator — orchestra i loader, lo scorer e la produzione delle 5 figure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Palette colori fissa USA / Cina
# ---------------------------------------------------------------------------
COLOR_USA   = "#1565C0"   # blu profondo
COLOR_CHINA = "#C62828"   # rosso profondo
FILL_USA    = "#90CAF9"   # azzurro chiaro per bande CI
FILL_CHINA  = "#EF9A9A"   # rosa chiaro per bande CI
COLOR_NEUTRAL = "#546E7A" # grigio blu per elementi neutri


# ===========================================================================
# 1. LOADER
# ===========================================================================
class EnergyRaceLoader:
    """
    Carica un file .npz prodotto da codice_5_export.py e ne espone i contenuti
    come attributi numpy con nomi leggibili.

    Esempio:
        usa = EnergyRaceLoader("risultati_UnitedStates.npz")
        usa.scenario_quota_fossile  # array (25,)
        usa.paese                   # "United States"
    """

    # Mappa nome_attributo -> chiave_npz (coincidono, ma esplicita il contratto)
    _SCALAR_KEYS = [
        "intensita_conflitto", "n_simulazioni", "mediana_emissioni_cum",
    ]
    _ARRAY_KEYS = [
        "anni_storici", "anni_futuri",
        "quota_fossile_storico", "quota_rinnovabile_storico", "quota_nucleare_storico",
        "consumo_totale_storico", "intensita_storica", "dipendenza_storica",
        "dip_verde_storica", "energia_per_capita_storica",
        "scenario_quota_fossile", "scenario_quota_nucleare", "scenario_quota_rinnovabile",
        "scenario_consumo", "scenario_intensita", "scenario_dipendenza",
        "scenario_dip_verde", "scenario_emissioni", "scenario_elettrificazione",
        "scenario_capex_pil", "scenario_capacita_nuc", "scenario_smr_capacity",
        "scenario_uranio", "scenario_prod_petrolio", "scenario_export_gnl",
        "energia_per_capita_futura",
        "consumo_fossile_twh", "consumo_nucleare_twh", "consumo_rinnovabile_twh",
        "sqf_low", "sqf_high", "sqn_low", "sqn_high", "sqr_low", "sqr_high",
        "sc_low", "sc_high", "em_low", "em_high",
        "nuc_cap_low", "nuc_cap_high", "ura_low", "ura_high", "pet_low", "pet_high",
        "mc_emissioni_cum",
        "mc_quota_fossile_p05", "mc_quota_fossile_p95",
        "mc_capacita_nuc_p05", "mc_capacita_nuc_p95",
        "mc_emissioni_p05", "mc_emissioni_p95",
        "mc_rinnovabile_p05", "mc_rinnovabile_p95",
        "radar_valori_2026", "radar_valori_2050",
        "bs_registro_keys", "bs_registro_valori",
    ]

    def __init__(self, path: str):
        p = self._resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"File non trovato: {p}\n"
                f"Esegui prima codice_5_export.py con il paese corrispondente."
            )
        raw = np.load(p, allow_pickle=True)

        # Paese come stringa
        self.paese = str(raw["paese"][0])
        self.path  = str(p)

        # Carica tutti gli array
        for key in self._ARRAY_KEYS:
            if key in raw:
                setattr(self, key, raw[key])
            else:
                setattr(self, key, None)
                # Non è fatale: alcune chiavi sono opzionali

        # Scalari
        self.intensita_conflitto = float(raw["intensita_conflitto"][0])
        self.n_simulazioni       = int(raw["n_simulazioni"][0])

        # Emissioni cumulative mediane (calcolate qui se non nel npz)
        self.mediana_emissioni_cum = float(np.median(self.mc_emissioni_cum)) if self.mc_emissioni_cum is not None else np.nan

        # Black Swan registry come dict
        if self.bs_registro_keys is not None and self.bs_registro_valori is not None:
            self.bs_registro = {
                str(k): int(v)
                for k, v in zip(self.bs_registro_keys, self.bs_registro_valori)
            }
        else:
            self.bs_registro = {}

        print(f"[LOADER] {self.paese} caricato da {p.name} "
              f"({self.n_simulazioni} run MC, conflitto={self.intensita_conflitto:.2f})")

    @staticmethod
    def _resolve_path(path: str) -> Path:
        """Risolvi percorsi relativi e versioni alternative del nome file NPZ."""
        source = Path(path)
        candidates = [source]

        if source.name == "risultati_UnitedStates.npz":
            candidates.append(source.with_name("risultati_United_States.npz"))
        elif source.name == "risultati_United_States.npz":
            candidates.append(source.with_name("risultati_UnitedStates.npz"))

        if source.name.startswith("risultati_") and " " in source.name:
            candidates.append(source.with_name(source.name.replace(" ", "_")))

        base_dirs = [Path.cwd(), Path(__file__).resolve().parent]
        base_dirs += list(Path(__file__).resolve().parents)[:4]

        for base in base_dirs:
            for candidate in candidates:
                candidate_path = candidate if candidate.is_absolute() else (base / candidate.name)
                if candidate_path.exists():
                    return candidate_path

        return source

    def anni_futuri_arr(self) -> np.ndarray:
        return self.anni_futuri

    def anni_storici_arr(self) -> np.ndarray:
        return self.anni_storici


# ===========================================================================
# 2. SCORER
# ===========================================================================
class EnergyRaceScorer:
    """
    Calcola le 10 metriche normalizzate [0-100] per la scorecard comparativa.

    Il punteggio 100 = massima performance; 0 = peggiore.
    Ogni metrica ha una direzione: 'higher_better' o 'lower_better'.

    Le metriche vengono normalizzate rispetto al range [min(USA,CN), max(USA,CN)]
    per rendere il confronto relativo tra le due superpotenze, non assoluto.
    Questo è coerente con l'obiettivo del modello: "chi vince la gara", non
    "chi raggiunge un benchmark esterno".
    """

    # ---------------------------------------------------------------------------
    # METRICHE — formato: (chiave, label, direzione, descrizione, range_assoluto, peso)
    #
    # range_assoluto = (min_fisico, max_fisico): scala il punteggio su valori reali
    #   invece di min-max relativo tra i 2 paesi (che produce sempre 0/100).
    #   Es: capacità nucleare va da 0 a 500 GW → un paese a 280 prende 56, non 0 o 100.
    #   Fonte range: IAEA, IEA, EIA 2024 projections per il 2050.
    #
    # peso = importanza relativa nella scorecard finale (somma = 1.0)
    #   Logica pesi:
    #   - Emissioni e Indipendenza pesano di più: sono gli obiettivi primari della "gara"
    #   - Nucleare e Rinnovabile: il come si arriva all'obiettivo
    #   - Efficienza, Elettrificazione, SMR: indicatori strutturali secondari
    #   - Sforzo economico e Uranio: vincoli di percorso, non obiettivi finali
    # ---------------------------------------------------------------------------
    METRICHE = [
        # (chiave, label, direzione, descrizione, (min,max), peso)
        ("quota_rin_2050",   "Quota\nRinnovabile",     "higher_better",
         "% non-fossile nel mix 2050",               (0.0,  100.0), 0.12),
        ("capacita_nuc_2050","Capacità\nNucleare",     "higher_better",
         "GW nucleari installati al 2050",           (0.0,  500.0), 0.10),
        ("emissioni_cum",    "Emissioni\nCumulative",  "lower_better",
         "Gt CO2 totali 2026-2050",                  (0.0,  400.0), 0.18),
        ("indip_geopolitica","Indipendenza\nGeopolit.","higher_better",
         "Indice composito indipendenza energetica", (0.0,  100.0), 0.18),
        ("intensita_2050",   "Efficienza\nEnergetica", "lower_better",
         "Intensità energetica TWh/mld$ al 2050",    (0.3,    3.5), 0.10),
        ("elettrificazione", "Elettrif-\nicazione",    "higher_better",
         "% elettrificazione finale al 2050",        (20.0, 100.0), 0.08),
        ("capex_sforzo",     "Sforzo\nEconomico",      "lower_better",
         "CAPEX/PIL medio 2026-2050",                (0.0,   10.0), 0.07),
        ("dip_uranio_2033",  "Indip.\nUranio 2033",    "lower_better",
         "% dipendenza uranio estero al 2033",       (0.0,  100.0), 0.07),
        ("fossil_exit_speed","Velocità\nUscita",       "higher_better",
         "pp/anno riduzione fossili",                (0.0,    5.0), 0.05),
        ("smr_2040",         "SMR\nDeployment",        "higher_better",
         "GW SMR installati al 2040",                (0.0,  150.0), 0.05),
    ]
    # Verifica che i pesi sommino a 1.0
    _PESO_TOTALE = sum(m[5] for m in METRICHE)
    assert abs(_PESO_TOTALE - 1.0) < 1e-6, f"Pesi non sommano a 1.0: {_PESO_TOTALE}"

    def __init__(self, loader_a: EnergyRaceLoader, loader_b: EnergyRaceLoader):
        self.a = loader_a
        self.b = loader_b
        self._raw: dict = {}      # valori grezzi {paese: {chiave: valore}}
        self._scores: dict = {}   # punteggi normalizzati [0-100]
        self._compute_raw()
        self._normalize()

    def _val(self, loader: EnergyRaceLoader, chiave: str) -> float:
        """Estrae il valore grezzo per una metrica da un loader."""
        af = loader.anni_futuri

        if chiave == "quota_rin_2050":
            return float(100.0 - loader.scenario_quota_fossile[-1])

        elif chiave == "capacita_nuc_2050":
            return float(loader.scenario_capacita_nuc[-1])

        elif chiave == "emissioni_cum":
            return float(loader.mediana_emissioni_cum / 1000)  # Gt

        elif chiave == "indip_geopolitica":
            # INDIPENDENZA GEOPOLITICA ENERGETICA — indice composito a 4 componenti
            #
            # La semplice inversione del net-import % non cattura la realtà USA:
            # gli USA sono esportatori netti di petrolio e LNG dal 2019, il che non è
            # solo "indipendenza" ma "dominanza" — usano l'energia come leva geopolitica.
            #
            # Componenti (tutte normalizzate 0-100, peso uguale):
            #
            # C1. Net energy position: esportatore netto = alta indipendenza
            #     Scala: da -100% import (totale dipendenza) a +50% export (dominanza)
            #     Formula: (valore + 100) / 150 * 100, clippato [0,100]
            _net_pos = float(np.clip(loader.scenario_dipendenza[-1], -50, 100))
            c1 = np.clip((-_net_pos + 100) / 150 * 100, 0, 100)

            # C2. Indipendenza uranio: 100 - dipendenza_uranio_2033
            #     Cattura la sicurezza del combustibile nucleare (HALEU, arricchimento domestico)
            _ura = float(loader.scenario_uranio[
                np.where(loader.anni_futuri == 2033)[0][0]
            ] if len(np.where(loader.anni_futuri == 2033)[0]) else loader.scenario_uranio[-1])
            c2 = np.clip(100 - _ura, 0, 100)

            # C3. Posizione export energetico: produzione petrolio + GNL come proxy di leva geopolitica
            #     Gli USA con 13.6 Mb/g e 50 Gm3 GNL sono dominanti; la Cina non esporta.
            #     Scala: prod_petrolio / 15 Mb/g (max plausibile) * 50% + export_gnl / 50 Gm3 * 50%
            _prod_pet = float(loader.scenario_prod_petrolio[-1]) if loader.scenario_prod_petrolio is not None else 0.0
            _exp_gnl  = float(loader.scenario_export_gnl[-1])    if loader.scenario_export_gnl is not None else 0.0
            c3 = np.clip((_prod_pet / 15.0) * 50 + (_exp_gnl / 50.0) * 50, 0, 100)

            # C4. Diversificazione del mix: meno dipendenza da una singola fonte = più resilienza
            #     Indice di Herfindahl-Hirschman del mix energetico (meno concentrato = più resiliente)
            _f = loader.scenario_quota_fossile[-1] / 100
            _n = loader.scenario_quota_nucleare[-1] / 100
            _r = loader.scenario_quota_rinnovabile[-1] / 100
            # Normalizza per sicurezza
            _tot = max(_f + _n + _r, 1e-6)
            _f, _n, _r = _f/_tot, _n/_tot, _r/_tot
            _hhi = _f**2 + _n**2 + _r**2   # range [0.33, 1.0]: 0.33 = perfetta diversif.
            c4 = np.clip((1.0 - _hhi) / (1.0 - 1/3) * 100, 0, 100)

            # Indice composito: pesi C1=35% (posizione netta), C2=25% (uranio),
            #                            C3=25% (leva export), C4=15% (diversificazione)
            # Penalità ME: uranio >70% dipendente + conflitto attivo = vulnerabilità supply chain
            _intensita_me = float(loader.intensita_conflitto) if hasattr(loader, 'intensita_conflitto') else 0.65
            _ura_penalty = float(np.clip((_ura - 70) / 30 * _intensita_me * 10, 0, 10))
            indip = 0.35 * c1 + 0.25 * c2 + 0.25 * c3 + 0.15 * c4 - _ura_penalty
            return float(np.clip(indip, 0, 100))

        elif chiave == "intensita_2050":
            return float(loader.scenario_intensita[-1])

        elif chiave == "elettrificazione":
            return float(loader.scenario_elettrificazione[-1])

        elif chiave == "capex_sforzo":
            return float(np.mean(loader.scenario_capex_pil))

        elif chiave == "dip_uranio_2033":
            idx = np.where(af == 2033)[0]
            return float(loader.scenario_uranio[idx[0]]) if len(idx) else float(loader.scenario_uranio[-1])

        elif chiave == "fossil_exit_speed":
            # pp/anno di riduzione fossile: media della derivata annua negativa
            delta = np.diff(loader.scenario_quota_fossile)
            return float(-np.mean(delta[delta < 0]))  # positivo = riduzione rapida

        elif chiave == "smr_2040":
            idx = np.where(af == 2040)[0]
            if loader.scenario_smr_capacity is None:
                return 0.0
            return float(loader.scenario_smr_capacity[idx[0]]) if len(idx) else float(loader.scenario_smr_capacity[-1])

        return float("nan")

    def _compute_raw(self):
        for loader in (self.a, self.b):
            self._raw[loader.paese] = {
                chiave: self._val(loader, chiave)
                for chiave, *_ in self.METRICHE
            }

    def _normalize(self):
        """
        Normalizzazione su range assoluto fisicamente motivato.

        Invece di min-max relativo tra i 2 paesi (che produce sempre 0/100),
        ogni metrica è scalata sul suo range fisico reale (min_fisico, max_fisico)
        definito in METRICHE[5]. Questo produce punteggi intermedi significativi:
        un paese a 280 GW su una scala 0-500 GW prende 56, non 0 o 100.

        higher_better: score = (val - min_fis) / (max_fis - min_fis) * 100
        lower_better:  score = (max_fis - val) / (max_fis - min_fis) * 100
        Clipping a [0, 100] se il valore grezzo esce dal range atteso.
        """
        for chiave, label, direzione, descr, rng_abs, peso in self.METRICHE:
            va = self._raw[self.a.paese][chiave]
            vb = self._raw[self.b.paese][chiave]
            vmin_abs, vmax_abs = rng_abs
            span = max(vmax_abs - vmin_abs, 1e-6)

            for loader, val in [(self.a, va), (self.b, vb)]:
                if direzione == "higher_better":
                    score = (val - vmin_abs) / span * 100
                else:
                    score = (vmax_abs - val) / span * 100
                # Clip: un valore fuori range ottiene 0 o 100, mai negativo
                score = float(np.clip(score, 0.0, 100.0))
                self._scores.setdefault(loader.paese, {})[chiave] = round(score, 1)

    def punteggi(self, paese: str) -> list:
        """Restituisce i punteggi nell'ordine di METRICHE."""
        return [self._scores[paese][m[0]] for m in self.METRICHE]

    def valori_grezzi(self, paese: str) -> list:
        return [self._raw[paese][m[0]] for m in self.METRICHE]

    def vincitore_metrica(self, chiave: str) -> str:
        """
        Restituisce il paese con punteggio più alto su una metrica.
        Con range assoluto, un gap < 8 punti è statisticamente trascurabile
        (< 8% della scala 0-100) e viene classificato come pareggio.
        """
        sa = self._scores[self.a.paese][chiave]
        sb = self._scores[self.b.paese][chiave]
        if abs(sa - sb) < 8:
            return "pareggio"
        return self.a.paese if sa > sb else self.b.paese

    def punteggio_totale(self, paese: str) -> float:
        """Media pesata con i pesi definiti in METRICHE[5]."""
        scores = self.punteggi(paese)
        pesi   = [m[5] for m in self.METRICHE]
        return float(np.average(scores, weights=pesi))

    def gap_metrica(self, chiave: str) -> float:
        """Differenza score USA - China su una metrica (positivo = USA avanti)."""
        sa = self._scores[self.a.paese][chiave]
        sb = self._scores[self.b.paese][chiave]
        return round(sa - sb, 1)

    def report_testuale(self) -> str:
        lines = [
            "=" * 62,
            f"  SCORECARD FINALE — LA CORSA ALL'ENERGIA",
            f"  {self.a.paese}  vs  {self.b.paese}",
            "=" * 62,
        ]
        for chiave, label, direzione, descr, rng_abs, peso in self.METRICHE:
            sa = self._scores[self.a.paese][chiave]
            sb = self._scores[self.b.paese][chiave]
            va = self._raw[self.a.paese][chiave]
            vb = self._raw[self.b.paese][chiave]
            win = self.vincitore_metrica(chiave)
            win_str = "=" if win == "pareggio" else ("USA" if win == self.a.paese else "CN")
            lines.append(
                f"  {label.replace(chr(10),' '):22s}  "
                f"USA {sa:5.1f}  CN {sb:5.1f}  {win_str}"
                f"  [{va:.2f} vs {vb:.2f}]"
            )
        tot_usa = self.punteggio_totale(self.a.paese)
        tot_cn  = self.punteggio_totale(self.b.paese)
        winner_str = self.a.paese if tot_usa >= tot_cn else self.b.paese
        lines += [
            "-" * 62,
            f"  TOTALE PESATO   USA {tot_usa:5.1f} / 100",
            f"                  CN  {tot_cn:5.1f} / 100",
            f"  VINCITORE: {winner_str}",
            "=" * 62,
        ]
        return "\n".join(lines)


# ===========================================================================
# 3. COMPARATOR
# ===========================================================================
class EnergyRaceComparator:
    """
    Orchestra il caricamento dei dati, il calcolo dei punteggi e la produzione
    delle 5 figure comparative.

    Figure prodotte:
        fig5a — Race chart: capacità nucleare nel tempo (con CI)
        fig5b — Fossil exit: quota fossile sovrapposta (con CI + crossover)
        fig5c — Radar doppio USA vs Cina 2050 (5 dimensioni)
        fig5d — Scorecard heatmap (10 metriche)
        fig5e — Emissioni cumulative: distribuzioni MC sovrapposte

    Uso:
        comp = EnergyRaceComparator("risultati_UnitedStates.npz", "risultati_China.npz")
        comp.run()
    """

    def __init__(self, path_usa: str, path_china: str):
        self.usa   = EnergyRaceLoader(path_usa)
        self.china = EnergyRaceLoader(path_china)
        self.scorer = EnergyRaceScorer(self.usa, self.china)
        print(self.scorer.report_testuale())

    # ------------------------------------------------------------------
    # Helper interni
    # ------------------------------------------------------------------
    @staticmethod
    def _fill_ci(ax, x, low, high, color, alpha=0.18, label=None):
        ax.fill_between(x, low, high, color=color, alpha=alpha, label=label)

    @staticmethod
    def _crossover(x, ya, yb):
        """
        Trova l'anno approssimativo in cui ya e yb si incrociano.
        Restituisce (anno, valore) o None se non si incrociano.
        """
        diff = ya - yb
        for i in range(len(diff) - 1):
            if diff[i] * diff[i+1] < 0:
                # Interpolazione lineare
                frac = abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1]))
                anno = x[i] + frac * (x[i+1] - x[i])
                valore = ya[i] + frac * (ya[i+1] - ya[i])
                return (anno, valore)
        return None

    # ------------------------------------------------------------------
    # Fig 5a — Race chart nucleare
    # ------------------------------------------------------------------
    def _fig_nuclear_race(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")

        af = self.usa.anni_futuri

        for loader, color, fill, label in [
            (self.usa,   COLOR_USA,   FILL_USA,   "United States"),
            (self.china, COLOR_CHINA, FILL_CHINA, "China"),
        ]:
            ax.plot(af, loader.scenario_capacita_nuc, color=color, lw=2.5,
                    label=f"{label} — Mediana")
            self._fill_ci(ax, af, loader.nuc_cap_low, loader.nuc_cap_high,
                          fill, alpha=0.20, label=f"{label} CI 95%")
            ax.plot(af, loader.mc_capacita_nuc_p05, color=color, lw=0.8,
                    ls=":", alpha=0.5)
            ax.plot(af, loader.mc_capacita_nuc_p95, color=color, lw=0.8,
                    ls=":", alpha=0.5)

        # Milestones orizzontali
        for gw, ls in [(200, "--"), (300, "-."), (400, "-")]:
            ax.axhline(gw, color="white", ls=ls, lw=0.7, alpha=0.3)
            ax.text(2026.2, gw + 3, f"{gw} GW", color="white", fontsize=8, alpha=0.5)

        # Crossover
        cx = self._crossover(af,
                             self.usa.scenario_capacita_nuc,
                             self.china.scenario_capacita_nuc)
        if cx:
            ax.axvline(cx[0], color="yellow", ls="--", lw=1.2, alpha=0.7)
            ax.text(cx[0] + 0.3, cx[1] + 8,
                    f"Crossover ~{int(cx[0])}",
                    color="yellow", fontsize=9, fontweight="bold")

        ax.set_title("NUCLEAR RACE — Capacità Installata (GW) 2026-2050",
                     color="white", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Anno", color="white")
        ax.set_ylabel("GW Nucleare Installato", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white",
                  loc="upper left")
        ax.grid(alpha=0.12, color="white")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Fig 5b — Fossil exit race
    # ------------------------------------------------------------------
    def _fig_fossil_exit(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0D1117")

        for ax in axes:
            ax.set_facecolor("#0D1117")

        af = self.usa.anni_futuri

        # Pannello sinistro: quota fossile assoluta sovrapposta
        ax = axes[0]
        for loader, color, fill, label in [
            (self.usa,   COLOR_USA,   FILL_USA,   "United States"),
            (self.china, COLOR_CHINA, FILL_CHINA, "China"),
        ]:
            # Storico
            ax.plot(loader.anni_storici, loader.quota_fossile_storico,
                    color=color, lw=1.5, ls="--", alpha=0.5)
            # Proiezione
            ax.plot(af, loader.scenario_quota_fossile, color=color, lw=2.5,
                    label=f"{label}")
            self._fill_ci(ax, af, loader.sqf_low, loader.sqf_high,
                          fill, alpha=0.18)

        # Crossover fossile
        cx = self._crossover(af,
                             self.usa.scenario_quota_fossile,
                             self.china.scenario_quota_fossile)
        if cx:
            ax.axvline(cx[0], color="yellow", ls="--", lw=1.2, alpha=0.8)
            ax.annotate(f"Crossover\n~{int(cx[0])}",
                        xy=(cx[0], cx[1]),
                        xytext=(cx[0] + 1.5, cx[1] + 5),
                        color="yellow", fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=1.0))

        ax.axvline(2026, color="#888", ls=":", lw=1, alpha=0.5)
        ax.set_title("Fossil Exit Race — Quota Fossile %",
                     color="white", fontsize=12, fontweight="bold")
        ax.set_ylabel("% Fossile nel Mix", color="white")
        ax.set_ylim(0, 100)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
        ax.grid(alpha=0.12, color="white")

        # Pannello destro: gap annuo (USA - China) nella quota fossile
        ax2 = axes[1]
        gap = self.usa.scenario_quota_fossile - self.china.scenario_quota_fossile
        gap_p05 = self.usa.mc_quota_fossile_p05 - self.china.mc_quota_fossile_p95
        gap_p95 = self.usa.mc_quota_fossile_p95 - self.china.mc_quota_fossile_p05

        # Colore del gap: positivo = USA più fossile (svantaggio USA), negativo = China più fossile
        ax2.fill_between(af, gap, 0,
                         where=(gap >= 0), color=FILL_CHINA, alpha=0.4,
                         label="Cina avanti nella transizione")
        ax2.fill_between(af, gap, 0,
                         where=(gap < 0), color=FILL_USA, alpha=0.4,
                         label="USA avanti nella transizione")
        ax2.fill_between(af, gap_p05, gap_p95, color=COLOR_NEUTRAL, alpha=0.12,
                         label="Range incertezza incrociato")
        ax2.plot(af, gap, color="white", lw=2)
        ax2.axhline(0, color="yellow", ls="--", lw=1, alpha=0.7)
        ax2.axvline(2026, color="#888", ls=":", lw=1, alpha=0.5)
        ax2.set_title("Gap Quota Fossile (USA − China) pp",
                     color="white", fontsize=12, fontweight="bold")
        ax2.set_ylabel("pp (USA − China)", color="white")
        ax2.tick_params(colors="white")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#444")
        ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        ax2.grid(alpha=0.12, color="white")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Fig 5c — Radar doppio 2050
    # ------------------------------------------------------------------
    def _fig_radar(self) -> plt.Figure:
        # Figura più larga per dare spazio alle etichette fuori dal cerchio
        fig = plt.figure(figsize=(13, 10))
        fig.patch.set_facecolor("#0D1117")

        # GridSpec: pannello radar a sinistra (più grande), legenda/testo a destra
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               width_ratios=[2.2, 1],
                               height_ratios=[1, 0.12],
                               hspace=0.08, wspace=0.05)
        ax = fig.add_subplot(gs[0, 0], polar=True)
        ax.set_facecolor("#0D1117")

        # Etichette corte (senza newline) per evitare troncature nel polar renderer
        # Le etichette vengono posizionate manualmente fuori dal cerchio
        categorie_corte = [m[1].replace(chr(10), " ") for m in self.scorer.METRICHE]
        N = len(self.scorer.METRICHE)

        # Array numpy per angoli (fix bug: lista Python -> np.array per polar)
        angoli = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angoli_chiusi = np.concatenate([angoli, [angoli[0]]])

        scores_usa_raw   = self.scorer.punteggi(self.usa.paese)
        scores_china_raw = self.scorer.punteggi(self.china.paese)
        scores_usa   = np.concatenate([scores_usa_raw,   [scores_usa_raw[0]]])
        scores_china = np.concatenate([scores_china_raw, [scores_china_raw[0]]])

        # Griglia di riferimento con etichette percentuale
        for r, alpha in [(25, 0.15), (50, 0.20), (75, 0.25), (100, 0.30)]:
            ax.plot(angoli_chiusi, [r] * (N + 1), color="#666", lw=0.6, alpha=alpha)
            ax.text(np.pi / 2, r + 2, f"{r}", color="#888", fontsize=6,
                    ha="center", va="bottom")

        # Raggi radiali
        for ang in angoli:
            ax.plot([ang, ang], [0, 100], color="#555", lw=0.4, alpha=0.4)

        # USA
        ax.plot(angoli_chiusi, scores_usa, color=COLOR_USA, lw=2.8,
                label="United States 2050", zorder=3)
        ax.fill(angoli_chiusi, scores_usa, color=COLOR_USA, alpha=0.18, zorder=2)

        # China
        ax.plot(angoli_chiusi, scores_china, color=COLOR_CHINA, lw=2.8,
                label="China 2050", zorder=3)
        ax.fill(angoli_chiusi, scores_china, color=COLOR_CHINA, alpha=0.18, zorder=2)

        # Punti sui vertici per leggibilità
        ax.scatter(angoli, scores_usa_raw,   s=45, color=COLOR_USA,   zorder=4)
        ax.scatter(angoli, scores_china_raw, s=45, color=COLOR_CHINA, zorder=4)

        # Etichette degli assi posizionate manualmente fuori dal cerchio (r=115)
        # Questo evita il bug di matplotlib polar che taglia o sovrappone le etichette
        ax.set_xticks([])          # disabilita etichette automatiche
        ax.set_yticks([])
        _r_label = 115             # raggio etichette
        for i, (ang, cat) in enumerate(zip(angoli, categorie_corte)):
            # Allineamento orizzontale dipende dalla posizione angolare
            ha = "center"
            if ang < 0.1 or ang > 2 * np.pi - 0.1:
                ha = "center"
            elif 0 < ang < np.pi:
                ha = "left" if ang < np.pi / 2 or ang > 3 * np.pi / 2 else "right"
                ha = "left" if np.sin(ang) > 0.1 else ("right" if np.sin(ang) < -0.1 else "center")
            ax.text(ang, _r_label, cat,
                    ha=ha, va="center",
                    color="white", fontsize=8.5, fontweight="bold")

        ax.set_ylim(0, 120)        # extra spazio per le etichette manuali
        ax.spines["polar"].set_color("#333")
        ax.set_facecolor("#0D1117")

        # Pannello testo destra: punteggi e vincitore
        ax_txt = fig.add_subplot(gs[0, 1])
        ax_txt.set_facecolor("#0D1117")
        ax_txt.axis("off")

        tot_usa   = self.scorer.punteggio_totale(self.usa.paese)
        tot_china = self.scorer.punteggio_totale(self.china.paese)
        winner = self.usa.paese if tot_usa >= tot_china else self.china.paese
        winner_color = COLOR_USA if winner == self.usa.paese else COLOR_CHINA

        # Legenda
        ax_txt.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.85), 0.9, 0.12, transform=ax_txt.transAxes,
            boxstyle="round,pad=0.02", facecolor="#1a1a2e", edgecolor=COLOR_USA, lw=1.5))
        ax_txt.text(0.5, 0.91, "🇺🇸  United States",
                    transform=ax_txt.transAxes, ha="center", color=COLOR_USA,
                    fontsize=11, fontweight="bold")

        ax_txt.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.70), 0.9, 0.12, transform=ax_txt.transAxes,
            boxstyle="round,pad=0.02", facecolor="#1a1a2e", edgecolor=COLOR_CHINA, lw=1.5))
        ax_txt.text(0.5, 0.76, "🇨🇳  China",
                    transform=ax_txt.transAxes, ha="center", color=COLOR_CHINA,
                    fontsize=11, fontweight="bold")

        # Tabella punteggi per metrica
        ax_txt.text(0.5, 0.63, "Punteggi per metrica",
                    transform=ax_txt.transAxes, ha="center", color="#aaa", fontsize=8)
        for i, m in enumerate(self.scorer.METRICHE):
            chiave = m[0]
            label  = m[1].replace(chr(10), " ")
            su = self.scorer._scores[self.usa.paese][chiave]
            sc = self.scorer._scores[self.china.paese][chiave]
            win_c = COLOR_USA if su > sc + 5 else (COLOR_CHINA if sc > su + 5 else "#aaa")
            y_pos = 0.57 - i * 0.055
            ax_txt.text(0.04, y_pos, label[:18],
                        transform=ax_txt.transAxes, color="#ccc", fontsize=7, va="center")
            ax_txt.text(0.70, y_pos, f"{su:.0f}",
                        transform=ax_txt.transAxes, color=COLOR_USA,
                        fontsize=7.5, fontweight="bold", ha="center", va="center")
            ax_txt.text(0.88, y_pos, f"{sc:.0f}",
                        transform=ax_txt.transAxes, color=COLOR_CHINA,
                        fontsize=7.5, fontweight="bold", ha="center", va="center")
            # Indicatore vincitore
            if win_c != "#aaa":
                sym = "◀" if win_c == COLOR_USA else "▶"
                ax_txt.text(0.79, y_pos, sym, transform=ax_txt.transAxes,
                            color=win_c, fontsize=7, ha="center", va="center")

        # Totali e vincitore
        ax_txt.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.00), 0.96, 0.08, transform=ax_txt.transAxes,
            boxstyle="round,pad=0.02", facecolor="#0a0a1a", edgecolor=winner_color, lw=2))
        ax_txt.text(0.5, 0.04,
                    f"USA {tot_usa:.1f}  vs  CN {tot_china:.1f}  →  🏆 {winner.split()[0]}",
                    transform=ax_txt.transAxes, ha="center", color=winner_color,
                    fontsize=10, fontweight="bold", va="center")

        # Titolo figura
        ax_title = fig.add_subplot(gs[1, :])
        ax_title.set_facecolor("#0D1117")
        ax_title.axis("off")
        ax_title.text(0.5, 0.5,
                      "ENERGY RACE 2050 — Scorecard Comparativa (punteggi normalizzati 0-100, min-max relativo USA vs China)",
                      transform=ax_title.transAxes, ha="center", color="#888",
                      fontsize=9, style="italic")

        return fig

    # ------------------------------------------------------------------
    # Fig 5d — Heatmap scorecard
    # ------------------------------------------------------------------
    def _fig_heatmap(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")

        n_met = len(self.scorer.METRICHE)
        etichette = [m[1].replace("\n", " ") for m in self.scorer.METRICHE]
        direzioni = [m[2] for m in self.scorer.METRICHE]

        sc_usa   = self.scorer.punteggi(self.usa.paese)
        sc_china = self.scorer.punteggi(self.china.paese)
        vr_usa   = self.scorer.valori_grezzi(self.usa.paese)
        vr_china = self.scorer.valori_grezzi(self.china.paese)

        # Costruzione matrice: righe = metriche, colonne = [USA_score, China_score]
        matrix = np.array([sc_usa, sc_china]).T  # (n_met, 2)

        # Colormap divergente: rosso scuro (basso) → bianco → verde scuro (alto)
        cmap = LinearSegmentedColormap.from_list(
            "race", ["#B71C1C", "#F8BBD9", "#F5F5F5", "#C8E6C9", "#1B5E20"]
        )
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

        # Etichette celle
        for i in range(n_met):
            for j, (sc, vr) in enumerate([(sc_usa[i], vr_usa[i]),
                                           (sc_china[i], vr_china[i])]):
                text_color = "black" if 30 < sc < 80 else "white"
                ax.text(j, i, f"{sc:.0f}",
                        ha="center", va="center",
                        fontsize=12, fontweight="bold", color=text_color)
                ax.text(j, i + 0.32, f"({vr:.1f})",
                        ha="center", va="center",
                        fontsize=7, color=text_color, alpha=0.7)

        # Assi
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["🇺🇸  United States", "🇨🇳  China"],
                           color="white", fontsize=13, fontweight="bold")
        pesi = [m[5] for m in self.scorer.METRICHE]
        ax.set_yticks(range(n_met))
        ax.set_yticklabels(
            [f"{'↑' if direzioni[i] == 'higher_better' else '↓'}  {etichette[i]}  "
             f"[peso {pesi[i]:.0%}]"
             for i in range(n_met)],
            color="white", fontsize=8.5
        )
        ax.xaxis.tick_top()

        # Riga vincitore evidenziata
        for i in range(n_met):
            win = self.scorer.vincitore_metrica(self.scorer.METRICHE[i][0])
            if win != "pareggio":
                j_win = 0 if win == self.usa.paese else 1
                rect = plt.Rectangle((j_win - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor="yellow",
                                     linewidth=2.0)
                ax.add_patch(rect)

        # Totali in fondo
        tot_usa   = self.scorer.punteggio_totale(self.usa.paese)
        tot_china = self.scorer.punteggio_totale(self.china.paese)
        ax.axhline(n_met - 0.5, color="#555", lw=1.5)

        # Barra visiva del gap pesato (proporzionale alla differenza di punteggio)
        gap_abs = abs(tot_usa - tot_china)
        winner_j = 0 if tot_usa >= tot_china else 1
        loser_j  = 1 - winner_j
        winner_score = max(tot_usa, tot_china)
        loser_score  = min(tot_usa, tot_china)

        # Rettangolo gap evidenziato sul vincitore
        ax.add_patch(plt.Rectangle(
            (winner_j - 0.48, n_met - 0.45), 0.96, 0.42,
            transform=ax.transData,
            facecolor=COLOR_USA if winner_j == 0 else COLOR_CHINA,
            alpha=0.25, zorder=0
        ))

        ax.text(winner_j, n_met + 0.05,
                f"🏆  {winner_score:.1f} / 100",
                ha="center", va="bottom", color="gold",
                fontsize=12, fontweight="bold", transform=ax.transData)
        ax.text(loser_j, n_met + 0.05,
                f"{loser_score:.1f} / 100  (−{gap_abs:.1f})",
                ha="center", va="bottom", color="#aaa",
                fontsize=10, transform=ax.transData)

        # Annotazione metodologia
        ax.text(0.5, -0.5,
                f"Punteggi su range assoluto fisicamente motivato  |  "
                f"Totale = media pesata (pesi in etichetta)",
                ha="center", va="center", color="#666",
                fontsize=7.5, style="italic", transform=ax.transData)

        ax.set_title(
            "SCORECARD DETTAGLIATA — 10 Metriche Energetiche al 2050\n"
            "(score 0-100 normalizzato, bordo giallo = vincitore metrica)",
            color="white", fontsize=12, fontweight="bold", pad=15
        )
        ax.spines[:].set_color("#444")
        plt.colorbar(im, ax=ax, label="Score (0=peggio, 100=meglio)",
                     fraction=0.02, pad=0.04)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Fig 5e — Distribuzioni emissioni cumulative sovrapposte
    # ------------------------------------------------------------------
    def _fig_emissions_dist(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle("EMISSIONI CUMULATIVE 2026-2050 (Gt CO2) — Distribuzione MC",
                     color="white", fontsize=13, fontweight="bold")

        for ax in axes:
            ax.set_facecolor("#0D1117")

        em_usa   = self.usa.mc_emissioni_cum   / 1000
        em_china = self.china.mc_emissioni_cum / 1000

        # Pannello sinistro: istogrammi sovrapposti
        ax = axes[0]
        bins = np.linspace(min(em_usa.min(), em_china.min()),
                           max(em_usa.max(), em_china.max()), 50)

        ax.hist(em_usa,   bins=bins, color=COLOR_USA,   alpha=0.55,
                label="United States", edgecolor="white", lw=0.3, density=True)
        ax.hist(em_china, bins=bins, color=COLOR_CHINA, alpha=0.55,
                label="China",         edgecolor="white", lw=0.3, density=True)

        for em, color, paese in [
            (em_usa,   COLOR_USA,   "USA"),
            (em_china, COLOR_CHINA, "CN"),
        ]:
            med = np.median(em)
            ax.axvline(med, color=color, lw=2, ls="--")
            ax.text(med + 0.05, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.1,
                    f"Mediana\n{paese}: {med:.1f} Gt",
                    color=color, fontsize=8)

        ax.set_xlabel("Gt CO2", color="white")
        ax.set_ylabel("Densità", color="white")
        ax.set_title("Distribuzioni sovrapposte", color="white", fontsize=11)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
        ax.grid(alpha=0.12, color="white")

        # Pannello destro: boxplot comparativo + annotazione probabilità
        ax2 = axes[1]
        bp = ax2.boxplot(
            [em_usa, em_china],
            labels=["United States", "China"],
            patch_artist=True,
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(color="white"),
            capprops=dict(color="white"),
            flierprops=dict(marker="o", color="gray", alpha=0.3, markersize=3),
        )
        bp["boxes"][0].set_facecolor(COLOR_USA)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(COLOR_CHINA)
        bp["boxes"][1].set_alpha(0.7)

        # Probabilità che USA < China (USA emette meno)
        p_usa_vince = float(np.mean(em_usa < em_china)) * 100
        ax2.text(0.5, 0.92,
                 f"P(USA emette meno di China) = {p_usa_vince:.1f}%",
                 transform=ax2.transAxes, ha="center", color="white",
                 fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round", fc="#1a1a2e", ec="white", alpha=0.8))

        ax2.set_ylabel("Gt CO2", color="white")
        ax2.set_title("Boxplot + probabilità", color="white", fontsize=11)
        ax2.tick_params(colors="white")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#444")
        ax2.grid(alpha=0.12, color="white", axis="y")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Orchestratore principale
    # ------------------------------------------------------------------
    def run(self, output_dir: str = "."):
        """
        Genera e salva tutte e 5 le figure comparative.
        """
        out = Path(output_dir)
        out.mkdir(exist_ok=True)

        print("\n[COMPARATOR] Generazione figure comparative...")

        figures = [
            ("fig5a_nuclear_race.png",   self._fig_nuclear_race,    "Nuclear Race"),
            ("fig5b_fossil_exit.png",    self._fig_fossil_exit,     "Fossil Exit"),
            ("fig5c_radar.png",          self._fig_radar,           "Radar Scorecard"),
            ("fig5d_heatmap.png",        self._fig_heatmap,         "Heatmap Scorecard"),
            ("fig5e_emissions_dist.png", self._fig_emissions_dist,  "Emissioni Distribution"),
        ]

        saved = []
        for fname, builder, label in figures:
            print(f"  Generando {label}...", end=" ")
            fig = builder()
            path = out / fname
            fig.savefig(path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            saved.append(str(path))
            print(f"[OK] {fname}")

        print(f"\n[COMPARATOR] {len(saved)} figure salvate in '{output_dir}/'")
        print("  " + "\n  ".join(saved))

        plt.show()
        return saved


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    import sys

    # Percorsi default (stessa cartella dello script)
    path_usa   = "risultati_United_States.npz"
    path_china = "risultati_China.npz"

    # Override da riga di comando: python codice_6_comparison.py usa.npz china.npz
    if len(sys.argv) == 3:
        path_usa, path_china = sys.argv[1], sys.argv[2]

    comp = EnergyRaceComparator(path_usa, path_china)
    comp.run(output_dir=".")
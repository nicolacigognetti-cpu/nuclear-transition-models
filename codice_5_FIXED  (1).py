import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import beta as beta_dist
import warnings
warnings.filterwarnings("ignore")

print("[DEBUG] *** VERSIONE CORRETTA - build 2026-06-06 ***")
print("[DEBUG] Fix attivi: USA rin k=0.04, fossil 85%, China uranium floor 42%, radar norm")

# ==========================================
# 1. CONFIGURAZIONE PROFILI GEOPOLITICI E STRATEGICI (Dati 2026)
# ==========================================

PROFILI_STATI = {
    "United States": {
        "pil_2023_mld": 27360,
        "crescita_pil": np.linspace(0.021, 0.015, 25), # 25 anni dal 2026 al 2050
        "ia_anno": 2024, 
        "ia_impatto_twh": 800, 
        "ia_fossile_bump": 5,
        "nuc_bat_anno": 2028, 
        "nuc_bat_impatto": -8,   # Ridotto da -30: in scenario Trump il nucleare non sostituisce fossili rapidamente.
                                  # Rimpiazza principalmente il carbone in declino naturale (40%->9% già avvenuto).
                                  # I fossili (gas soprattutto) restano dominanti come baseload fino al 2040+.
        "nuc_bat_dip_fossile": -5,  # Ridotto da -15: stessa logica
        # CARBON TAX USA — realtà politica aggiornata al 2026:
        # L'amministrazione Trump ha smantellato ogni politica climatica (IRA ridimensionato,
        # zero emissioni 2050 cancellato). Una carbon tax federale nel 2030 è incompatibile
        # con la traiettoria politica attuale.
        # Modelliamo invece uno shock condizionale: con probabilità p_policy_reversal
        # (calcolata nel MC in funzione dell'anno elettorale 2028/2032) si attiva una
        # politica climatica post-Trump. L'impatto è più contenuto (-15, non -40) perché
        # anche un'amministrazione democratica non partirebbe da zero con una carbon tax piena.
        # p_policy_reversal e l'anno di attivazione sono campionati nel loop MC.
        "carbon_tax_anno": 2033,       # anno più probabile: post-elezioni 2032 + 1 anno implementazione
        "carbon_tax_impatto": -12,     # impatto ridotto: politica parziale, non Green New Deal completo
        "carbon_tax_p_attivazione": 0.35,  # 35% di probabilità che avvenga entro il 2050
        # Nota: lo shock viene attivato stocasticamente nel MC (non deterministicamente come prima)
        "green_dependency_start": 70, 
        "green_independence_speed": -1.5,
        "crescita_popolazione": np.linspace(1.0, 1.12, 25),
        "me_shock_anno": 2027, 
        "me_shock_durata": 4, 
        "me_acceleratore_verde": 0.5, 
        "me_fossil_price_bump": 1.1,
        "max_rinnovabili": 47,
        "tanbreez_anno": 2029, 
        "tanbreez_impatto_export": -45,
        
        # --- NUOVI PARAMETRI STRATEGICI USA 2026 ---
        "quota_fossile_2026": None,          # Sincronizzato dinamicamente con i dati storici
        "quota_nucleare_2026": None,         # Sincronizzato dinamicamente con i dati storici
        "prod_petrolio_2026": 13.6,          # Produzione petrolio greggio USA iniziale a 13.6 Mb/g (23% dell'offerta globale)
        "export_gnl_2026": 25.0,             # Esportazione iniziale di GNL in Gm3
        "capacita_nucleare_2026": 100.0,     # Base di partenza capacità nucleare a 100 GW
        "dipendenza_uranio_2026": 95.0,      # Dipendenza iniziale dall'uranio estero al 95%
        "obiettivo_nucleare_2050": 400.0,     # Obiettivo capacità nucleare di 400 GW entro il 2050
        "emission_factor": 0.23              # Calibrato: ~4,800 Mt CO2 reali per ~21,000 TWh fossili
    },
    "China": {
        # === PARAMETRI MACROECONOMICI ===
        "pil_2023_mld": 17700,
        "crescita_pil": np.linspace(0.045, 0.020, 25),

        # === IMPATTO IA ===
        # Domanda IA cinese forte ma più distribuita su infrastruttura coal/rinnovabili
        "ia_anno": 2026, "ia_impatto_twh": 600, "ia_fossile_bump": 8,

        # === NUCLEARE ===
        # Fonte: il piano cinese punta a 400 GW nucleari entro 2050 (stesso obiettivo USA).
        # Tempi di costruzione cinesi: 5-7 anni vs decenni occidentali. LCOE ~2800$/kW vs 9000$/kW USA.
        # Stato finanzia il 71.2% → nessun rischio investitore privato → costruzione continua senza stop.
        # SMR: vantaggio stimato 10-15 anni sugli USA. Linglong One (ACP100) e HTR-PM già avanzati.
        # Obiettivo intermedio: peak emissioni prima del 2030, neutralità carbonica 2060.
        "nuc_bat_anno": 2025,
        "nuc_bat_impatto": -45,        # Impatto più forte (-45 vs -40) per costruzione statale continua
        "nuc_bat_dip_fossile": -30,    # Riduzione fossili più aggressiva grazie a HTR (calore + H2)

        # === CARBON POLICY ===
        # La Cina non ha una carbon tax forte nel breve, ma ha un ETS (sistema quote CO2) già attivo.
        # Peak emissioni prima del 2030; post-2030 politica energetica più restrittiva sul carbone.
        "carbon_tax_anno": 2030,       # Anticipato: picco emissioni fissato pre-2030, poi stretta
        "carbon_tax_impatto": -15,     # Leggero (-15): il carbone è ancora strategico fino al 2035+

        # === DIPENDENZE VERDI ===
        # La Cina è esportatrice netta di tecnologie rinnovabili (81% celle FV, 50.91% eolico mondiale).
        # Ha catena di fornitura interna integrata → dipendenza verde estera NEGATIVA (esportatrice).
        "green_dependency_start": -30,
        "green_independence_speed": 0,

        # === DEMOGRAFIA ===
        "crescita_popolazione": np.linspace(1.0, 0.82, 25),

        # === SHOCK MEDIO ORIENTE ===
        # La Cina importa ~75% del petrolio via mare; tensioni nello Stretto di Hormuz la colpiscono
        # più degli USA ma accelerano anche la transizione interna verso rinnovabili + nucleare.
        "me_shock_anno": 2027, "me_shock_durata": 4,
        "me_acceleratore_verde": 2.5,  # Forte acceleratore verde: ogni crisi fossile spinge il rinnov.
        "me_fossil_price_bump": 1.3,

        # === RINNOVABILI ===
        # 35% energia solare mondiale, 50.91% eolico globale, ~1/3 idroelettrico mondiale.
        # Potenziale solare teorico 20 milioni TWh. Rete HVDC per trasporto deserti→coste.
        # La Cina raggiungerà obiettivi climatici 2030 già nel 2025.
        "max_rinnovabili": 90,         # Alzato a 90: dominanza globale su solare, eolico, idro

        # === BATTERIE E SOLID-STATE ===
        "ss_battery_anno": 2026, "ss_efficiency_floor": 0.52, "ss_elec_boost": 8,  # 52% floor: Cina 2026, cresce a 65-70% al 2050

        # === PARAMETRI STRATEGICI CINA (Fonte: PDF credito libero informazioni) ===
        "quota_fossile_2026": None,    # Sincronizzato dinamicamente con dati OWID (carbone ~60% mix)
        "quota_nucleare_2026": None,   # Sincronizzato dinamicamente con dati OWID (~5% mix attuale)
        "prod_petrolio_2026": 4.0,     # Mb/g: la Cina è produttrice modesta, importa ~75% del fabbisogno
        "export_gnl_2026": 0.0,        # Nessun export GNL rilevante
        "capacita_nucleare_2026": 55.0,  # Aggiornato: ~50 reattori operativi fine 2020 + nuovi dal 2021
        "dipendenza_uranio_2026": 66.0,  # Ridotto a 66%: realtà attuale ~70-75%, floor 42% al 2050 è credibile
        "obiettivo_nucleare_2050": 400.0, # Aggiornato a 400 GW: stesso ambizioso target USA (fonte PDF)

        # === COSTI COSTRUZIONE E CAPEX NUCLEARE ===
        # LCOE cinese ~2800$/kW (vs 9000$/kW USA, 6600$/kW EU). Tempi 5-7 anni.
        # Stato copre costi di capitale → nessun rischio mercato → costruzione continua.
        "lcoe_vantaggio_factor": 0.31,  # 2800/9000 ≈ 0.31 → costo costruzione ~3x inferiore agli USA

        # === FUSIONE NUCLEARE ===
        # EAST operativo 2006, plasma stabile 1066s (gen 2025), 120M°C raggiunti.
        # CFETR: costruzione 2030, operativo ~2040, obiettivo 200MW→1GW.
        # 1.5 mld$/anno investimento pubblico; China Fusion Energy Co. (2.1 mld$) lanciata 2025.
        # 34% quota mondiale IP fusione; spesa 6.5 mld$ nel periodo 2023-2025.
        "fusione_ricerca_anno": 2030,   # CFETR: inizio costruzione target
        "fusione_commerciale_anno": 2055, # PFPP: centrale commerciale nel decennio 2050-2060

        # === CARBONE (PARADOSSO CINESE) ===
        # ~60% mix elettrico da carbone; 97 GW carbone in costruzione; 152 GW in progetto.
        # Non abbandonerà il carbone nel breve periodo → baseline fossile alta fino al ~2035.
        "carbone_peak_anno": 2030,      # Peak carbone post-2030 in linea con impegni climatici
        "carbone_exit_rate": 0.025,     # Tasso di uscita lento: 2.5% annuo post-peak (struttura pesante)

        # === IDROELETTRICO (Diga Tre Gole + espansione) ===
        # Cina: 1/3 produzione idroelettrica mondiale. 50% espansione idro globale ultimi 15 anni.
        # Diga Tre Gole: 22.5 GW (la più potente al mondo).
        "idro_quota_attuale": 16.0,     # Idroelettrico ~16% produzione elettrica cinese

        "emission_factor": 0.30        # Calibrato: ~11,000 Mt CO2 reali (alto per dominanza carbone)
    },
    "European Union (27)": {
        "pil_2023_mld": 18350,
        "crescita_pil": np.linspace(0.005, 0.012, 25),
        "ia_anno": 2026, "ia_impatto_twh": 300, "ia_fossile_bump": 2,
        "nuc_bat_anno": 2028, "nuc_bat_impatto": -35, "nuc_bat_dip_fossile": -20,
        "carbon_tax_anno": 2026, "carbon_tax_impatto": -25,
        "green_dependency_start": 90, 
        "green_independence_speed": -0.8,
        "crescita_popolazione": np.linspace(1.0, 0.96, 25),
        "me_shock_anno": 2027, "me_shock_durata": 4, "me_acceleratore_verde": 4.0, "me_fossil_price_bump": 2.0,
        "max_rinnovabili": 75,
        "plastic_h2_anno": 2027, "plastic_capex_discount": 0.30, "plastic_indip_bonus": -20,
        
        # Defaults per uniformità
        "quota_fossile_2026": None,
        "quota_nucleare_2026": None,
        "prod_petrolio_2026": 0.5,
        "export_gnl_2026": 0.0,
        "capacita_nucleare_2026": 115.0,
        "dipendenza_uranio_2026": 90.0,
        "obiettivo_nucleare_2050": 150.0,
        "emission_factor": 0.19              # Calibrato: ~2,600 Mt CO2 reali
    }
}

# --- SELEZIONE INTERATTIVA DEL PAESE ---
_PAESI_DISPONIBILI = list(PROFILI_STATI.keys())
print("\n=== SELEZIONE PAESE ===")
print("Paesi disponibili:")
for idx, p in enumerate(_PAESI_DISPONIBILI, 1):
    print(f"  {idx}. {p}")
try:
    _scelta = input(f"Inserisci il numero o il nome del paese [default: 1 = United States]: ").strip()
    if _scelta == "" or _scelta == "1":
        PAESE = "United States"
    elif _scelta.isdigit() and 1 <= int(_scelta) <= len(_PAESI_DISPONIBILI):
        PAESE = _PAESI_DISPONIBILI[int(_scelta) - 1]
    elif _scelta in PROFILI_STATI:
        PAESE = _scelta
    else:
        print(f"  Valore non riconosciuto, uso default: United States")
        PAESE = "United States"
except Exception:
    PAESE = "United States"
print(f"-> Paese selezionato: {PAESE}\n")

PROFILO = PROFILI_STATI.get(PAESE, PROFILI_STATI["United States"])

# --- COSTANTI DI SCENARIO (IA) ---
# Moltiplicatore calibrato a 1.25X per mantenere un impatto finale di ~300-500 TWh al 2050 con efficienza progressiva
AI_MULTIPLIER = 1.25

# --- INTRODUZIONE DELLA VARIABILE GEOPOLITICA (Input robusto non bloccante) ---
print("=== IMPOSTAZIONE SCENARIO GEOPOLITICO ===")
try:
    val_in = input("Inserisci l'intensità media del conflitto in Medio Oriente (float da 0.0 a 1.0, dove 1.0 = blocco Stretto di Hormuz) [default 0.65]: ").strip()
    intensita_conflitto_me = float(val_in) if val_in else 0.65
except Exception:
    intensita_conflitto_me = 0.65

# Clip per garantire che resti nell'intervallo richiesto [0.0, 1.0]
intensita_conflitto_me = np.clip(intensita_conflitto_me, 0.0, 1.0)
print(f"-> Intensità media del conflitto impostata a: {intensita_conflitto_me:.2f}\n")

# ==========================================
# 2. CARICAMENTO E PREPARAZIONE DATI STORICI
# ==========================================
# Mappa paese → codici API
_CODICI_WB   = {"United States": "US", "China": "CN", "European Union (27)": "EU"}
_WB_CODE     = _CODICI_WB.get(PAESE, "US")

# ------------------------------------------------------------------
# 2a. WORLD BANK API — PIL reale (USD correnti) 1990-2024
# Indicatore NY.GDP.MKTP.CD: https://datahelpdesk.worldbank.org/
# Restituisce una pd.Series vuota in caso di errore di rete.
# ------------------------------------------------------------------
def _fetch_wb_gdp(iso2, start=1990, end=2024):
    import urllib.request, json
    url = (f"https://api.worldbank.org/v2/country/{iso2}/indicator/NY.GDP.MKTP.CD"
           f"?format=json&date={start}:{end}&per_page=100")
    try:
        with urllib.request.urlopen(url, timeout=12) as resp:
            raw = json.loads(resp.read().decode())
        records = raw[1] if (isinstance(raw, list) and len(raw) > 1) else []
        data = {int(r["date"]): float(r["value"]) / 1e9
                for r in records if r.get("value") is not None}
        s = pd.Series(data).sort_index()
        print(f"  [WB OK] PIL scaricato: {len(s)} anni, "
              f"{int(s.iloc[0])} mld$ ({s.index[0]}) -> {int(s.iloc[-1])} mld$ ({s.index[-1]})")
        return s
    except Exception as e:
        print(f"  [WB FALLBACK] Impossibile scaricare PIL ({e}).")
        return pd.Series(dtype=float)

print(f"Scaricando dati World Bank (PIL) per {PAESE} [{_WB_CODE}]...")
_serie_pil_wb = _fetch_wb_gdp(_WB_CODE)

# ------------------------------------------------------------------
# 2b. OWID — dati energetici
# ------------------------------------------------------------------
print(f"Scaricando dati OWID per {PAESE}...")
url_dati = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
df_mondo = pd.read_csv(url_dati)
df_paese = df_mondo[df_mondo['country'] == PAESE].copy()
df_storico = df_paese[(df_paese['year'] >= 1990) & (df_paese['year'] <= 2025)].copy()
anni_storici = df_storico['year'].values

def _safe(df, col):
    if col in df.columns:
        return df[col].interpolate(method='linear').bfill().ffill().values
    return None

consumo_totale        = _safe(df_storico, 'primary_energy_consumption')
quota_fossile         = _safe(df_storico, 'fossil_share_energy')
quota_rinnovabile     = _safe(df_storico, 'renewables_share_energy')
quota_nucleare_storica = _safe(df_storico, 'nuclear_share_energy')
if quota_nucleare_storica is None:
    quota_nucleare_storica = np.clip(100.0 - quota_fossile - quota_rinnovabile, 0.0, 100.0)

pop_arr = _safe(df_storico, 'population')
if pop_arr is None or np.any(np.isnan(pop_arr)):
    _pb  = {"United States": 338e6, "China": 1410e6}.get(PAESE, 448e6)
    _tp  = {"United States": 0.0065, "China": 0.003}.get(PAESE, 0.001)
    pop_arr = np.array([_pb * (1 + _tp) ** (a - 2025) for a in anni_storici])
popolazione_storica = pop_arr

# ------------------------------------------------------------------
# 2c. INTENSITA' ENERGETICA — TWh / miliardo USD (PIL reale WB)
# Valori tipici: USA ~2.5 TWh/mld$ nel 1990, ~1.0 nel 2020.
# ------------------------------------------------------------------
if len(_serie_pil_wb) >= 10:
    _pil_aligned = (_serie_pil_wb
                    .reindex(anni_storici)
                    .interpolate(method='index')
                    .bfill().ffill())
    _pil_arr = _pil_aligned.values
else:
    # Fallback 1: colonna 'gdp' di OWID (USD correnti)
    _owid_gdp = _safe(df_storico, 'gdp')
    if _owid_gdp is not None and not np.all(np.isnan(_owid_gdp)):
        _pil_arr = _owid_gdp / 1e9
        print("  [FALLBACK] PIL da OWID colonna 'gdp'.")
    else:
        # Fallback 2: estrapolazione con CAGR costante
        _cagr_fb = float(np.mean(PROFILO["crescita_pil"]))
        _b23 = PROFILO["pil_2023_mld"]
        _pil_arr = np.array([_b23 * (1 - _cagr_fb) ** (2023 - a) for a in anni_storici])
        print("  [FALLBACK] PIL estrapolato (CAGR costante).")

_pil_arr = np.clip(_pil_arr, 1.0, None)
intensita_storica = consumo_totale / _pil_arr   # TWh / mld USD

# ------------------------------------------------------------------
# 2d. DIPENDENZA FOSSILE GEOPOLITICA — net energy import % (OWID)
# Formula: (consumo_primario - produzione_interna) / consumo * 100
# Positivo = importatore netto (dipendente). Negativo = esportatore.
# ------------------------------------------------------------------
_prod_tot = _safe(df_storico, 'energy_production')
if _prod_tot is not None:
    dipendenza_storica = np.clip(
        (consumo_totale - _prod_tot) / np.clip(consumo_totale, 1e-6, None) * 100.0,
        -150.0, 150.0
    )
    print(f"  [OK] Dipendenza fossile (net import %): "
          f"{dipendenza_storica[0]:.1f}% (1990) -> {dipendenza_storica[-1]:.1f}% (recente)")
else:
    _t = -5.0 if PAESE == "United States" else (60.0 if PAESE == "European Union (27)" else 20.0)
    dipendenza_storica = np.linspace(30.0, _t, len(anni_storici))
    print("  [FALLBACK] Dipendenza fossile: linspace (produzione non in OWID).")

# ------------------------------------------------------------------
# 2e. DIPENDENZA MINERALI CRITICI VERDI — proxy ancorato ai dati OWID
# Principio: quanto piu' un paese e' indietro nella transizione rinnovabile
# rispetto al suo potenziale, tanto piu' importa tecnologie (pannelli, batterie,
# turbine) da fornitori esteri.
# Proxy = gap tra quota rinnovabile corrente e tetto teorico del paese.
# ------------------------------------------------------------------
_max_rin_teo = float(PROFILO.get("max_rinnovabili", 80))
_gap_rin = np.clip((_max_rin_teo - quota_rinnovabile) / _max_rin_teo * 100.0, 0.0, 100.0)
# Ancoriamo il valore finale al green_dependency_start del profilo
_shift_dv = PROFILO["green_dependency_start"] - _gap_rin[-1]
dip_verde_storica = _gap_rin + _shift_dv
print(f"  [OK] Dipendenza minerali critici (proxy gap rin.): "
      f"{dip_verde_storica[0]:.1f}% (1990) -> {dip_verde_storica[-1]:.1f}% (recente)")

energia_per_capita_storica = (consumo_totale * 1e6) / popolazione_storica  # MWh/persona

# ==========================================
# 2.5 SINCRONIZZAZIONE DINAMICA PARAMETRI STRATEGICI CON I DATI STORICI REALI (CSV)
# ==========================================
# Allineiamo automaticamente i parametri del profilo strategico agli ultimi valori del dataset storico
PROFILO["quota_fossile_2026"] = quota_fossile[-1]
PROFILO["quota_nucleare_2026"] = quota_nucleare_storica[-1]

# ==========================================
# 3. BASELINE MATEMATICA E PROIEZIONE 2026 - 2050
# ==========================================
anni_storici_ml = anni_storici.reshape(-1, 1)
# Orizzonte temporale: dal 2026 al 2050 (25 anni)
anni_futuri = np.arange(2026, 2051)
anni_futuri_ml = anni_futuri.reshape(-1, 1)

# Dinamiche Macro-economiche (PIL Reale) — baseline deterministica
# Questa è la traiettoria PIL in assenza di shock. Nel loop MC viene perturbata
# dagli eventi BSE (crisi finanziaria -> delta_pil < 0) per ogni run.
_pil_base_futuro = np.zeros(len(anni_futuri))
_pil_base_futuro[0] = PROFILO["pil_2023_mld"] * (1.02**3)  # rivalutato al 2026
for i in range(1, len(anni_futuri)):
    _pil_base_futuro[i] = _pil_base_futuro[i-1] * (1 + PROFILO["crescita_pil"][i-1])
# pil_futuro_mld rimane come alias per compatibilità con il codice esistente (es. grafici)
pil_futuro_mld = _pil_base_futuro.copy()

def modello_asintotico(t, a, b, c):
    return a * np.exp(-b * (t - anni_storici[0])) + c

def fit_baseline_asintotica(X, y, X_future):
    """
    Fit asintotico robusto per serie storiche, incluse serie che cambiano segno
    (es. net energy imports USA: da +25% a -10%).
    Strategia: prova il modello asintotico con bounds larghi; se fallisce usa
    un polinomio di grado 2 smorzato verso il valore finale (evita estrapolazione
    divergente tipica della regressione lineare pura).
    """
    t_hist = X.flatten()
    t_fut  = X_future.flatten()
    c_guess = y[-1]
    a_guess = y[0] - c_guess

    # Tentativo 1: fit asintotico standard (bounds larghi, accetta valori negativi)
    try:
        popt, _ = curve_fit(
            modello_asintotico, t_hist, y,
            p0=[a_guess, 0.05, c_guess],
            bounds=([-np.inf, 1e-4, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=8000
        )
        proj = modello_asintotico(t_fut, *popt)
        # Sanity check: la proiezione non deve divergere oltre 3x il range storico
        _range = max(abs(y.max() - y.min()), 1.0)
        if np.all(np.abs(proj - y[-1]) < 3 * _range):
            return proj
        raise ValueError("proiezione divergente")
    except Exception:
        pass

    # Tentativo 2: modello lineare con smorzamento verso il valore finale
    # La pendenza decade esponenzialmente (tau = 15 anni) per evitare estrapolazione illimitata
    slope = float(LinearRegression().fit(X[-10:], y[-10:]).coef_[0])  # pendenza sull'ultimo decennio
    proj = np.zeros(len(t_fut))
    current = y[-1]
    for k, t in enumerate(t_fut):
        dt = t - t_hist[-1]
        current = y[-1] + slope * dt * np.exp(-dt / 15.0)  # smorzamento tau=15 anni
        proj[k] = current
    return proj

def fit_baseline_per_capita(t_hist, y_hist, t_fut):
    def asintoto_saturazione(t, a, b, c):
        return a * np.exp(-b * (t - t_hist[0])) + c
    
    # Regressione lineare preliminare per determinare la direzione del trend storico
    modello_lineare = LinearRegression()
    modello_lineare.fit(t_hist.reshape(-1, 1), y_hist)
    slope = modello_lineare.coef_[0]
    
    try:
        if slope < 0:
            # Trend decrescente (es. USA, EU): decresce asintoticamente verso un limite inferiore di efficienza (c)
            c_guess = y_hist[-1] * 0.75
            a_guess = y_hist[0] - c_guess # a > 0 in modo che decada verso il basso
            popt, _ = curve_fit(
                asintoto_saturazione, t_hist, y_hist, 
                p0=[a_guess, 0.05, c_guess], 
                bounds=([0.0, 0.01, 0.0], [np.inf, 0.5, y_hist[-1]]), 
                maxfev=5000
            )
        else:
            # Trend crescente (es. Cina): cresce asintoticamente verso un limite di saturazione superiore (c).
            # c_guess moderato (1.15x invece di 1.3x) per evitare sovrastima della saturazione
            # che causerebbe un plateau troppo alto o troppo precoce nel grafico.
            c_guess = max(y_hist) * 1.15
            a_guess = y_hist[0] - c_guess  # a < 0 in modo che salga verso l'alto
            popt, _ = curve_fit(
                asintoto_saturazione, t_hist, y_hist, 
                p0=[a_guess, 0.05, c_guess], 
                bounds=([-np.inf, 0.01, max(y_hist)], [0.0, 0.5, 90000.0]), 
                maxfev=5000
            )
        return asintoto_saturazione(t_fut, *popt)
    except:
        # Fallback stazionario coerente che preserva la pendenza decrescente o crescente
        base = np.zeros(len(t_fut))
        current_y = y_hist[-1]
        decay_slope = slope
        for i in range(len(t_fut)):
            decay_slope *= np.exp(-0.08) # La pendenza sfuma lentamente nel tempo
            current_y += decay_slope
            base[i] = current_y
        return base


def modello_logistico(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

base_per_capita = fit_baseline_per_capita(anni_storici, energia_per_capita_storica, anni_futuri)
# Allineamento curve per continuità perfetta
base_per_capita = base_per_capita - (base_per_capita[0] - energia_per_capita_storica[-1])

max_rin = PROFILO.get("max_rinnovabili", 80)
try:
    popt_log, _ = curve_fit(modello_logistico, anni_storici, quota_rinnovabile, p0=[max_rin, 0.1, 2035], bounds=([0, 0, 1900], [max_rin + 0.1, 1.0, 2100]))
    base_quota_rinnovabile_logistica = modello_logistico(anni_futuri, *popt_log)
    base_quota_rinnovabile_logistica = base_quota_rinnovabile_logistica - (base_quota_rinnovabile_logistica[0] - quota_rinnovabile[-1])
except:
    quota_rinnovabile_10y_ago = quota_rinnovabile[-11]
    quota_rinnovabile_oggi = quota_rinnovabile[-1]
    cagr_rinnovabili = (quota_rinnovabile_oggi / quota_rinnovabile_10y_ago)**(1/10) - 1
    k_logistico = max(0.05, cagr_rinnovabili * 5) 
    base_quota_rinnovabile_logistica = max_rin / (1 + np.exp(-k_logistico * (anni_futuri - 2035)))
    base_quota_rinnovabile_logistica = base_quota_rinnovabile_logistica - (base_quota_rinnovabile_logistica[0] - quota_rinnovabile_oggi)

# USA POLICY OVERRIDE: curve_fit usa dati IRA 2022-2024, incompatibile con scenario Trump.
# k=0.04 = crescita lenta solo da mercato; inflection 2042 senza sussidi federali.
if PAESE == "United States":
    _q0_rin = float(quota_rinnovabile[-1])
    base_quota_rinnovabile_logistica = max_rin / (1.0 + np.exp(-0.04 * (anni_futuri - 2042)))
    base_quota_rinnovabile_logistica = base_quota_rinnovabile_logistica - (base_quota_rinnovabile_logistica[0] - _q0_rin)
    print(f"  [USA OVERRIDE rin] 2026={base_quota_rinnovabile_logistica[0]:.1f}% -> 2050={base_quota_rinnovabile_logistica[-1]:.1f}%")

# Baseline del nucleare: costante al valore storico (la crescita nucleare avviene nel MC loop)
base_quota_nucleare = np.full(len(anni_futuri), quota_nucleare_storica[-1])

# La baseline fossile completa il mix a 100%
base_quota_fossile_logistica = 100.0 - base_quota_rinnovabile_logistica - base_quota_nucleare

# --- USA FOSSIL BASELINE OVERRIDE ---
# Con k=0.04 per i rinnovabili, la baseline fossile scende comunque di ~11pp al 2050.
# Nel documento Trump vuole mantenere i fossili dominanti — la baseline deve essere quasi piatta.
# Forziamo un declino massimo di 5pp al 2050 (solo effetto carbone in calo naturale).
if PAESE == "United States":
    _fos_0 = float(quota_fossile[-1])
    _fos_target_2050 = _fos_0 - 5.0  # declino massimo 5pp in 25 anni
    base_quota_fossile_logistica = np.linspace(_fos_0, _fos_target_2050, len(anni_futuri))
    # Ricalcola rinnovabili come residuo per coerenza del mix baseline
    base_quota_rinnovabile_logistica = 100.0 - base_quota_fossile_logistica - base_quota_nucleare
    print(f"  [USA OVERRIDE fos] Baseline fossile: 2026={base_quota_fossile_logistica[0]:.1f}% -> 2050={base_quota_fossile_logistica[-1]:.1f}%")

# --- FIX DISCONTINUITÀ FOSSILE (solo per non-USA, già allineato nell'override sopra) ---
if PAESE != "United States":
    _offset_fossile = base_quota_fossile_logistica[0] - quota_fossile[-1]
    base_quota_fossile_logistica = base_quota_fossile_logistica - _offset_fossile

fattore_demografico = PROFILO["crescita_popolazione"]
popolazione_futura = popolazione_storica[-1] * fattore_demografico
base_consumo_totale = (base_per_capita * popolazione_futura) / 1e6

base_intensita = fit_baseline_asintotica(anni_storici_ml, intensita_storica, anni_futuri_ml)
base_intensita = base_intensita - (base_intensita[0] - intensita_storica[-1])

base_dipendenza = fit_baseline_asintotica(anni_storici_ml, dipendenza_storica, anni_futuri_ml)
base_dipendenza = base_dipendenza - (base_dipendenza[0] - dipendenza_storica[-1])

base_dip_verde = fit_baseline_asintotica(anni_storici_ml, dip_verde_storica, anni_futuri_ml)
base_dip_verde = base_dip_verde - (base_dip_verde[0] - dip_verde_storica[-1])

elettrificazione_base = 20 + 55 / (1 + np.exp(-0.2 * (anni_futuri - 2035)))

# ==========================================
# 4. MODELLAZIONE DEGLI SHOCK E DEI COMPORTAMENTI DISSOCIATI (Medio Oriente)
# ==========================================
def curva_logistica_shock(anni_futuri, anno_innesco, impatto_massimo, k=0.25, flesso=10):
    impatto = []
    for anno in anni_futuri:
        if anno <= anno_innesco:
            impatto.append(0)
        else:
            diffusione = impatto_massimo / (1 + np.exp(-k * ((anno - anno_innesco) - flesso)))
            impatto.append(diffusione)
    return np.array(impatto)

# Funzione per calcolare gli shock incrementali rispetto al 2025 (fine dello storico)
# Risolve l'anomalia dell'offset per shock che iniziano nel passato (come l'IA nel 2024)
def calcola_shock_incrementale(anni_futuri, anno_innesco, impatto_massimo, k=0.25, flesso=10):
    val_2025 = curva_logistica_shock([2025], anno_innesco, impatto_massimo, k, flesso)[0]
    shock_futuro = curva_logistica_shock(anni_futuri, anno_innesco, impatto_massimo, k, flesso)
    return shock_futuro - val_2025

# SHOCK CARBON TAX — gestione differenziata per paese
# USA: shock condizionale stocastico (probabilità da profilo, anno campionato vicino alle elezioni)
# Tutti gli altri paesi: shock deterministico come prima
if PAESE == "United States":
    # La baseline deterministica è ZERO: nessuna carbon tax garantita.
    # Nel loop MC ogni run campiona se e quando si attiva (vedi sezione 5).
    shock_carbon_tax = np.zeros(len(anni_futuri))
    _p_tax_usa = PROFILO.get("carbon_tax_p_attivazione", 0.35)
    _anno_tax_base = PROFILO.get("carbon_tax_anno", 2033)
    _impatto_tax = PROFILO.get("carbon_tax_impatto", -12)
    print(f"  [USA Carbon Tax] Shock stocastico: p={_p_tax_usa:.0%}, "
          f"anno base {_anno_tax_base}, impatto {_impatto_tax} pp")
else:
    shock_carbon_tax = calcola_shock_incrementale(
        anni_futuri, PROFILO["carbon_tax_anno"], PROFILO["carbon_tax_impatto"],
        k=0.8, flesso=3
    )

# Calcolo dell'impatto IA con moltiplicatore 1.25X ed efficienza tecnologica progressiva del 3.5% all'anno
# Risolve l'anomalia del multiplier eccessivo a 10x e limita l'impatto finale 2050 a ~300-500 TWh (USA: 400 TWh)
ia_consumo_shock = calcola_shock_incrementale(anni_futuri, PROFILO["ia_anno"], PROFILO["ia_impatto_twh"] * AI_MULTIPLIER, k=0.4, flesso=5)
ia_efficienza = np.exp(-0.035 * np.maximum(0, anni_futuri - PROFILO["ia_anno"]))
ia_consumo = ia_consumo_shock * ia_efficienza

ia_fossile = calcola_shock_incrementale(anni_futuri, PROFILO["ia_anno"], PROFILO["ia_fossile_bump"] * 0.1, k=0.3, flesso=5)
nuc_bat_fossile = calcola_shock_incrementale(anni_futuri, PROFILO["nuc_bat_anno"], PROFILO["nuc_bat_impatto"], k=0.25, flesso=8)
nuc_dipendenza = calcola_shock_incrementale(anni_futuri, PROFILO["nuc_bat_anno"], PROFILO["nuc_bat_dip_fossile"], k=0.25, flesso=8)

# Shock economico sul costo del capitale dovuto alle sanzioni e speculazioni di mercato

moltiplicatore_capex = np.ones(len(anni_futuri))
for idx, anno in enumerate(anni_futuri):
    if anno >= PROFILO["me_shock_anno"] and anno <= (PROFILO["me_shock_anno"] + PROFILO["me_shock_durata"]):
        anni_dallo_shock = anno - PROFILO["me_shock_anno"]
        picco_panico = PROFILO["me_fossil_price_bump"] * 1.5
        moltiplicatore_capex[idx] = 1.0 + (picco_panico - 1.0) * np.exp(-0.8 * anni_dallo_shock)

# Bolla speculativa centrata sul 2027 che gonfia temporaneamente i costi di installazione
bolla_green = 0.25 * np.exp(-((anni_futuri - 2027) ** 2) / (2 * 1.5 ** 2))
# Non sottrarre bolla_green[0]: l'allineamento creava valori negativi nelle code post-2030
# che facevano scendere il moltiplicatore sotto 1.0 (fisicamente impossibile: costi < baseline).
moltiplicatore_capex = moltiplicatore_capex + bolla_green
# Floor a 1.0: il moltiplicatore non può mai essere inferiore al costo base
moltiplicatore_capex = np.maximum(moltiplicatore_capex, 1.0)

# Drastica riduzione del feedback climatico a 5.0 TWh all'anno (risolve Anomalia 2)
feedback_climatico_twh = np.array([max(0, (anno - 2026) * 5.0) for anno in anni_futuri])


# ==========================================
# 4b. BLACK SWAN ENGINE
# ==========================================
# Cinque categorie di eventi estremi modellate con processo di Poisson.
# Le frequenze sono calibrate su dati storici osservati (1950-2024).
#
# Metodologia:
#   P(k eventi in T anni) = Poisson(lambda * T)
#   Per ogni run MC, campionamento di un processo di Poisson non-omogeneo
#   (la probabilita' cambia con il tempo per alcuni eventi).
#
# FONTI CALIBRAZIONE:
#   - Incidenti nucleari INES >= 5: IAEA PRIS database (2 eventi in 50 anni di
#     operatività globale, con ~450 reattori attivi) -> lambda ~ 0.04/reattore/anno
#     a livello di sistema-paese: ~1 evento ogni 15 anni per paese con 50+ GW
#   - Crisi finanziarie sistemiche (>= Lehman 2008): BIS database (5 crisi gravi
#     in 70 anni nelle economie avanzate) -> lambda ~ 1/14 per anno per paese
#   - Rottura tecnologica negativa (fallimento tecnologia energetica chiave):
#     stima esperta -> 10% probabilita' in 25 anni -> lambda ~ 0.004/anno
#   - Rottura tecnologica positiva (fusione, solare < 2 cent):
#     stima conservativa -> 15% probabilita' in 25 anni -> lambda ~ 0.006/anno
#   - Shock geopolitico extra-ME (Taiwan Strait, sanzioni, ecc.):
#     SIPRI conflict database -> lambda ~ 1/20 per anno

class BlackSwanEngine:
    """
    Gestisce l'estrazione e l'applicazione di eventi estremi per un singolo run MC.

    Attributi pubblici dopo .draw():
        eventi_attivi : dict  {categoria: lista di anni in cui l'evento e' attivo}
        n_eventi      : int   numero totale di eventi estratti
    """

    # --- PARAMETRI CALIBRATI SU DATI STORICI (modificabili per scenario) ---
    # Formato: (lambda_base, durata_anni, dizionario_impatti)
    # lambda_base = frequenza media eventi per anno (Poisson rate)
    # durata_anni = quanto dura l'effetto prima di smorzarsi
    # impatti = {variabile: delta_per_anno_durante_evento}
    CATALOGO = {
        # 1. INCIDENTE NUCLEARE (tipo Fukushima: INES >= 5)
        # Effetto: moratoria politica -> rallenta costruzione, riduce quota nucleare,
        # spinge temporaneamente fossili per compensare.
        # Fonte frequenza: IAEA PRIS, 2 eventi INES>=5 in 70 anni su ~450 reattori-anno globali
        #   -> per paese con capacita' media: ~1 evento ogni 20 anni
        "incidente_nucleare": {
            "lambda": 1 / 20,
            "durata": 8,       # l'effetto dura ~8 anni (vedi risposta post-Fukushima in Germania)
            "impatti": {
                "delta_nuc_quota":    -3.5,   # % mix nucleare perso per anno (moratoria + shutdown)
                "delta_fossile":      +2.5,   # % fossili compensano
                "delta_nuc_cap_rate": -0.6,   # moltiplicatore annuo sulla crescita capacita'
                "delta_fiducia_nuc":  -20.0,  # indice fiducia pubblica (non nel modello ma loggato)
            }
        },
        # 2. CRISI FINANZIARIA SISTEMICA (tipo 2008 o peggio)
        # Effetto: investimenti CAPEX crollano, costruzione rallenta, PIL cala.
        # Fonte: BIS (2023) "Banking crises: a 200-year historical overview" ->
        #   ~5 crisi gravi in 70 anni nelle economie G7 -> lambda ~ 1/14
        "crisi_finanziaria": {
            "lambda": 1 / 14,
            "durata": 4,       # effetto acuto ~4 anni, poi recupero
            "impatti": {
                "delta_capex_mult":   -0.35,  # investimenti rinnovabili -35% per anno della crisi
                "delta_pil":          -0.025, # PIL -2.5% per anno
                "delta_fossile":      +1.5,   # rinvio transizione -> piu' fossili nel breve
                "delta_rin_rate":     -0.8,   # rallenta crescita rinnovabili (% per anno)
                "delta_nuc_cap_rate": -0.4,   # FEEDBACK LOOP: crisi finanziaria -> meno investimenti nucleari
            }
        },
        # 3. ROTTURA TECNOLOGICA NEGATIVA
        # Es: batterie solid-state hanno problemi di sicurezza di scala non risolvibili
        #     entro il decennio; scoperta che i pannelli perovskite degradano in <5 anni.
        # Effetto: azzera o ritarda uno shock tecnologico gia' previsto.
        # Stima: ~12% probabilita' in 25 anni per almeno una tecnologia chiave
        "tech_breakdown": {
            "lambda": 0.005,
            "durata": 10,      # il ritardo tecnico dura ~10 anni
            "impatti": {
                "delta_rin_rate":     -1.2,   # rallenta crescita rinnovabili
                "delta_elec_boost":   -0.5,   # riduce elettrificazione
                "delta_max_rin":      -8.0,   # abbassa il tetto fisico delle rinnovabili
            }
        },
        # 4. BREAKTHROUGH TECNOLOGICO POSITIVO
        # Es: fusione commerciale ahead-of-schedule; solare a 1 cent/kWh entro 2035.
        # Effetto: accelera la transizione piu' del previsto.
        # Stima: ~18% probabilita' in 25 anni (ottimismo calibrato su ritmo innovazione
        #   solare 2010-2024: -90% costi in 14 anni, mai visto prima storicamente)
        "tech_breakthrough": {
            "lambda": 0.008,
            "durata": 15,      # l'effetto positivo si propaga per 15 anni
            "impatti": {
                "delta_rin_rate":     +2.5,   # accelera crescita rinnovabili
                "delta_fossile":      -3.0,   # accelera uscita fossile
                "delta_elec_boost":   +1.0,   # accelera elettrificazione
                "delta_max_rin":      +10.0,  # alza il tetto fisico (piu' tecnologie disponibili)
            }
        },
        # 5. SHOCK GEOPOLITICO NON-MEDIO ORIENTE
        # Es: crisi Taiwan Strait -> blocco semiconduttori -> stop costruzione reattori SMR;
        #     sanzioni energetiche -> blocco uranio; guerre commerciali sui minerali critici.
        # Fonte: SIPRI Military Expenditure Database + crisi Taiwan (2 episodi seri in 30 anni)
        #   -> lambda ~ 1/20 per paese
        "shock_geopolitico_extra": {
            "lambda": 1 / 20,
            "durata": 5,
            "impatti": {
                "delta_nuc_cap_rate": -0.3,   # rallenta costruzione reattori (supply chain)
                "delta_dip_verde":    +15.0,  # dipendenza minerali critici sale
                "delta_fossile":      +1.0,   # rerouting su fossili nel breve
            }
        },
    }

    def __init__(self, paese: str, anni_futuri: np.ndarray, seed: int = None):
        self.paese       = paese
        self.anni_futuri = anni_futuri
        self.n_anni      = len(anni_futuri)
        self.eventi_attivi = {k: [] for k in self.CATALOGO}
        self.n_eventi    = 0
        if seed is not None:
            np.random.seed(seed)

        # Fattori di scala per paese: alcuni eventi sono piu' probabili in certi contesti
        self._lambda_scale = self._get_lambda_scale(paese)

    def _get_lambda_scale(self, paese: str) -> dict:
        """
        Modulazione country-specific delle lambda.
        USA: piu' esposto a crisi finanziarie (mercato privato), meno a incidenti nucleari
             (regolamentazione NRC molto severa).
        China: piu' esposto a shock geopolitici Taiwan; meno a crisi finanziarie (controllo stato);
               costruzione nucleare intensa -> maggior esposizione statistica a incidenti.
        EU: alta sensibilita' politica agli incidenti nucleari (vedi Germania post-Fukushima).
        """
        scales = {
            "United States": {
                "incidente_nucleare": 0.6,      # NRC rigorosa
                "crisi_finanziaria": 1.3,       # mercato finanziario piu' volatile
                "tech_breakdown": 1.0,
                "tech_breakthrough": 1.2,       # ecosistema innovazione piu' dinamico
                "shock_geopolitico_extra": 0.9,
            },
            "China": {
                "incidente_nucleare": 0.65,     # ridotto da 1.4: reattori standardizzati (Hualong One) + regolamentazione centralizzata
                                               # target: 50-60% probabilità cumulativa in 25 anni vs 82% precedente
                "crisi_finanziaria": 0.5,       # stato controlla il sistema finanziario
                "tech_breakdown": 0.9,
                "tech_breakthrough": 1.1,
                "shock_geopolitico_extra": 1.6, # Taiwan Strait, sanzioni
            },
            "European Union (27)": {
                "incidente_nucleare": 0.8,
                "crisi_finanziaria": 1.1,
                "tech_breakdown": 1.0,
                "tech_breakthrough": 0.9,       # ecosistema innovazione piu' lento
                "shock_geopolitico_extra": 1.1,
            },
        }
        return scales.get(paese, {k: 1.0 for k in self.CATALOGO})

    def draw(self) -> "BlackSwanEngine":
        """
        Campiona il processo di Poisson per ogni categoria di evento.
        Popola self.eventi_attivi con gli anni in cui ogni evento e' in corso.
        Usa un processo non-omogeneo: la probabilita' di breakdown tecnologico
        diminuisce man mano che una tecnologia matura (effetto apprendimento).
        """
        self.eventi_attivi = {k: [] for k in self.CATALOGO}
        self.n_eventi = 0

        for categoria, spec in self.CATALOGO.items():
            lam_base = spec["lambda"] * self._lambda_scale.get(categoria, 1.0)
            durata   = spec["durata"]

            # Processo di Poisson: campiona i tempi inter-evento con distribuzione Esponenziale
            # Questo e' il metodo esatto (non approssimato) per simulare un processo di Poisson
            t_corrente = 0.0
            while True:
                # Tempo all'evento successivo ~ Exp(lambda)
                # Per tech_breakdown: lambda decade col tempo (le tecnologie diventano piu' mature)
                if categoria == "tech_breakdown":
                    lam_eff = lam_base * np.exp(-0.04 * t_corrente)  # decay 4% per anno
                elif categoria == "tech_breakthrough":
                    lam_eff = lam_base * (1 + 0.02 * t_corrente)    # lievemente crescente
                else:
                    lam_eff = lam_base

                lam_eff = max(lam_eff, 1e-6)
                wait = np.random.exponential(1.0 / lam_eff)
                t_corrente += wait

                if t_corrente > self.n_anni:
                    break

                # Anno di innesco (intero, relativo al 2026)
                idx_innesco = min(int(t_corrente), self.n_anni - 1)
                anno_innesco = self.anni_futuri[idx_innesco]

                # Marca tutti gli anni della durata come "attivi" per questo evento
                for d in range(durata):
                    idx_att = idx_innesco + d
                    if idx_att < self.n_anni:
                        self.eventi_attivi[categoria].append(self.anni_futuri[idx_att])

                self.n_eventi += 1

        return self

    def get_delta(self, categoria: str, anno: int) -> dict:
        """
        Restituisce il dizionario impatti per una data categoria e anno.
        Se l'anno non e' attivo, restituisce tutti zeri.
        L'impatto e' smorzato esponenzialmente con la durata dell'evento
        (picco nel primo anno, poi decadimento).
        """
        if anno not in self.eventi_attivi[categoria]:
            return {k: 0.0 for k in self.CATALOGO[categoria]["impatti"]}

        # Trova quanto siamo avanti nella durata dell'evento (per smorzamento)
        anni_att = sorted(self.eventi_attivi[categoria])
        # Identifica il primo anno di questo "episodio" (il piu' vicino precedente non in lista)
        idx_in_lista = [a for a in anni_att if a <= anno]
        # Stima posizione nella durata: 0 = primo anno (impatto pieno), cresce poi decade
        pos = len(idx_in_lista) - 1
        decay = np.exp(-0.25 * pos)  # decade del ~22% per anno dopo il picco

        return {k: v * decay for k, v in self.CATALOGO[categoria]["impatti"].items()}

    def summary(self) -> str:
        """Stringa riassuntiva degli eventi estratti per questo run."""
        lines_out = []
        for cat, anni in self.eventi_attivi.items():
            if anni:
                anni_unici = sorted(set(anni))
                lines_out.append(f"  {cat}: {anni_unici[0]}-{anni_unici[-1]}")
        return "\n".join(lines_out) if lines_out else "  (nessun evento)"

# --- Pre-allocazione registro eventi per statistiche post-MC ---
# Conta quanti run hanno avuto almeno un evento per categoria
registro_bs = {k: 0 for k in BlackSwanEngine.CATALOGO}

# ==========================================
# 5. SIMULAZIONE MONTE CARLO (YEAR-BY-YEAR)
# ==========================================
N_SIMULAZIONI = 1000

# Inizializzazione Array Monte Carlo
mc_quota_fossile = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_quota_nucleare = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_quota_rinnovabile = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_prod_petrolio = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_export_gnl = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_capacita_nucleare = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_smr_capacity = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_dipendenza_uranio = np.zeros((N_SIMULAZIONI, len(anni_futuri)))

mc_consumo = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_intensita = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_dipendenza = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_dip_verde = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_consumo_fossile_twh = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_consumo_nucleare_twh = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_consumo_rinnovabile_twh = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_energia_per_capita = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_elettrificazione = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_storage = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_emissioni = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_emissioni_cum = np.zeros(N_SIMULAZIONI)
mc_capex_pil = np.zeros((N_SIMULAZIONI, len(anni_futuri)))
mc_pil       = np.zeros((N_SIMULAZIONI, len(anni_futuri)))  # PIL stocastico per run

miglioramento_supply_chain = np.array([(anno - 2026) * PROFILO["green_independence_speed"] for anno in anni_futuri])

storico_fossile_twh = consumo_totale * (quota_fossile / 100)
storico_nucleare_twh = consumo_totale * (quota_nucleare_storica / 100)
storico_rinnovabile_twh = consumo_totale * (quota_rinnovabile / 100)

# Calcoliamo la deviazione standard per il rumore Monte Carlo delle rinnovabili (start a 0.5 per incertezza di base)
std_dev_array = np.linspace(0.5, 3.5, len(anni_futuri))

# Allineamento del decadimento della correlazione temporale (adattamento geopolitico)
anni_dal_conflitto = np.arange(len(anni_futuri))
rho_decay = 0.6472 * np.exp(-0.15 * anni_dal_conflitto)  # decade a ~0.15 dopo 10 anni

# Costruiamo la matrice di covarianza time-varying fuori dal loop per ottimizzare le prestazioni
cov_eta_t = np.zeros((len(anni_futuri), 2, 2))
for t in range(len(anni_futuri)):
    cov_eta_t[t] = [[1.0, rho_decay[t]], [rho_decay[t], 1.0]]

# Calcoliamo i parametri della distribuzione Beta per la variabilità stocastica del conflitto
mu_beta = intensita_conflitto_me
kappa_beta = 10.0
alpha_beta = mu_beta * kappa_beta
beta_beta = (1.0 - mu_beta) * kappa_beta
alpha_beta = max(0.1, alpha_beta)
beta_beta = max(0.1, beta_beta)

print("Esecuzione Simulazione Monte Carlo in corso...")
for i in range(N_SIMULAZIONI):
    # 1. Generazione del fattore geopolitico comune del run
    z_geo = np.random.normal(0, 1)
    
    # Mapping Copula di z_geo alla distribuzione Beta per l'intensità del conflitto
    p = 0.5 * (1.0 + erf(z_geo / np.sqrt(2.0)))
    intensita_run = beta_dist.ppf(p, alpha_beta, beta_beta)
    
    # 2a. BLACK SWAN ENGINE — campionamento eventi estremi per questo run
    bse = BlackSwanEngine(PAESE, anni_futuri).draw()
    for cat in bse.eventi_attivi:
        if bse.eventi_attivi[cat]:
            registro_bs[cat] += 1

    # 2b-ext. CARBON TAX USA — campionamento stocastico per questo run
    # Per gli USA la carbon tax non è garantita: viene attivata con probabilità p_tax_usa.
    # L'anno di attivazione è distribuito attorno al 2033 (post-elezioni 2032) con
    # una finestra di ±4 anni per catturare incertezza politica.
    # Per tutti gli altri paesi shock_carbon_tax è già deterministico (calcolato sopra).
    if PAESE == "United States":
        if np.random.rand() < _p_tax_usa:
            # Attivazione: campiona anno vicino alle elezioni (2028 o 2032 +1 anno implementazione)
            _anno_tax_run = int(np.random.choice([2029, 2033, 2037],
                                                  p=[0.20, 0.55, 0.25]))
            shock_carbon_tax_run = calcola_shock_incrementale(
                anni_futuri, _anno_tax_run, _impatto_tax, k=0.8, flesso=3
            )
        else:
            # Nessuna carbon tax in questo run (scenario Trump/continuità conservatrice)
            shock_carbon_tax_run = np.zeros(len(anni_futuri))
    else:
        shock_carbon_tax_run = shock_carbon_tax  # deterministico per Cina/EU

    # 2b. Rallentamento narrativo (scandalo rinnovabili) — mantenuto come evento separato
    freno_narrativo = np.zeros(len(anni_futuri))
    if np.random.rand() < 0.15:
        anno_scandalo = np.random.choice(anni_futuri[2:15])
        indice_scandalo = np.where(anni_futuri == anno_scandalo)[0][0]
        for j in range(indice_scandalo, min(indice_scandalo + 4, len(anni_futuri))):
            freno_narrativo[j] = 2.0  
            
    # 3. Generazione dei rumori con correlazione time-varying (decadimento temporale)
    eta = np.zeros((len(anni_futuri), 2))
    for t in range(len(anni_futuri)):
        eta[t] = np.random.multivariate_normal([0, 0], cov_eta_t[t])
        
    # 4. Composizione lineare per ottenere le correlazioni esatte
    z_mc = 0.5 * z_geo + np.sqrt(0.75) * eta[:, 0]
    z_dip = 0.35 * z_geo + np.sqrt(0.8775) * eta[:, 1]
    
    # 5. Applicazione delle deviazioni standard e del freno stocastico
    rumore_mc = z_mc * std_dev_array + freno_narrativo
    rumore_mc_dip = z_dip * (std_dev_array * 0.5)
    
    # Array per l'evoluzione annuale di questa specifica traiettoria
    quota_fossile_t = np.zeros(len(anni_futuri))
    quota_nucleare_t = np.zeros(len(anni_futuri))
    quota_rinnovabile_t = np.zeros(len(anni_futuri))
    prod_petrolio_t = np.zeros(len(anni_futuri))
    export_gnl_t = np.zeros(len(anni_futuri))
    capacita_nucleare_t = np.zeros(len(anni_futuri))
    smr_capacity_t = np.zeros(len(anni_futuri))
    dipendenza_uranio_t = np.zeros(len(anni_futuri))
    
    # --- PIL STOCASTICO PER QUESTO RUN ---
    # Partiamo dalla baseline e applichiamo i delta_pil degli eventi BSE.
    # Meccanismo: per ogni anno in cui è attiva una crisi finanziaria, il tasso di
    # crescita PIL di quell'anno viene ridotto di delta_pil (es. -2.5% -> crescita
    # effettiva = crescita_base - 0.025). L'effetto è cumulativo ma temporaneo:
    # finita la crisi, il PIL torna a crescere al tasso base (no hysteresis permanente,
    # scelta conservativa per evitare sovrastime dell'impatto a 25 anni).
    pil_run = np.zeros(len(anni_futuri))
    pil_run[0] = _pil_base_futuro[0]
    for _t_pil in range(1, len(anni_futuri)):
        _anno_pil = anni_futuri[_t_pil]
        _delta_pil_bse = bse.get_delta("crisi_finanziaria", _anno_pil).get("delta_pil", 0.0)
        # Il tasso di crescita base viene ridotto dalla crisi (es. 2.1% -> -0.4% netto)
        _tasso_eff = PROFILO["crescita_pil"][_t_pil - 1] + _delta_pil_bse
        pil_run[_t_pil] = pil_run[_t_pil - 1] * (1 + _tasso_eff)
    # Protezione: il PIL non può mai andare sotto il 50% del valore iniziale
    pil_run = np.clip(pil_run, _pil_base_futuro[0] * 0.5, None)

    # Assegnazione valori iniziali 2026 (continuità perfetta dallo storico)
    quota_fossile_t[0] = quota_fossile[-1]
    quota_nucleare_t[0] = quota_nucleare_storica[-1]
    quota_rinnovabile_t[0] = quota_rinnovabile[-1]
    prod_petrolio_t[0] = PROFILO["prod_petrolio_2026"]
    export_gnl_t[0] = PROFILO["export_gnl_2026"]
    capacita_nucleare_t[0] = PROFILO["capacita_nucleare_2026"]
    smr_capacity_t[0] = 0.0
    dipendenza_uranio_t[0] = PROFILO["dipendenza_uranio_2026"]
    
    # Inizializzazione variabili smorzamento per l'inerzia temporale
    
    # CICLO ANNUALE DI SIMULAZIONE (fino al 2050)
    for t in range(1, len(anni_futuri)):
        anno = anni_futuri[t]
        
        # 1. STRATEGIA "DRILL, BABY, DRILL" (FOSSILE)
        standard_decline = base_quota_fossile_logistica[t] - base_quota_fossile_logistica[t-1]
        # USA: 85% del declino annullato per politica Trump (doc: produttore n.1 mondiale).
        # Cina/EU: 30% * intensita conflitto ME.
        if PAESE == "United States":
            fossil_compensation = -standard_decline * (0.85 + intensita_run * 0.12)
        else:
            fossil_compensation = -standard_decline * intensita_run * 0.3
        
        # VARIAZIONI DEGLI SHOCK STRATEGICI (Carbon Tax, Carico IA, SMR micro-nucleari)
        # Ripristina i legami di feedback causale precedentemente orfani sul mix energetico!
        delta_carbon_tax = shock_carbon_tax_run[t] - shock_carbon_tax_run[t-1]
        delta_ia_fossile = ia_fossile[t] - ia_fossile[t-1]
        delta_nuc_bat = nuc_bat_fossile[t] - nuc_bat_fossile[t-1]
        
        # --- BLACK SWAN: impatti sulle quote del mix ---
        _bs_fossile  = bse.get_delta("incidente_nucleare", anno).get("delta_fossile", 0.0)
        _bs_fossile += bse.get_delta("crisi_finanziaria",  anno).get("delta_fossile", 0.0)
        _bs_fossile += bse.get_delta("shock_geopolitico_extra", anno).get("delta_fossile", 0.0)
        _bs_fossile -= bse.get_delta("tech_breakthrough", anno).get("delta_fossile", 0.0)  # nota: impatto negativo

        target_fossile = (quota_fossile_t[t-1] + standard_decline + fossil_compensation
                          + delta_carbon_tax + delta_ia_fossile + delta_nuc_bat
                          + rumore_mc[t] + _bs_fossile)
        
        # Smorzamento del mix energetico (damping = 0.65)
        quota_fossile_t[t] = quota_fossile_t[t-1] + 0.65 * (target_fossile - quota_fossile_t[t-1])
        quota_fossile_t[t] = np.clip(quota_fossile_t[t], 5.0, 100.0)
        
        # 2. PRODUZIONE DI PETROLIO GREGGIO (Mb/g) CON RAMPA SMORZATA
        if PAESE == "United States":
            # USA: "Drill, Baby, Drill" — target sale con il conflitto, declino terminale dal 2038
            prod_target = 13.6 + intensita_run * 2.0  # plateau tra 13.6 (neutro) e 15.6 Mb/g (conflitto pieno)
            if anno >= 2038:
                prod_target *= max(0.85, 1.0 - (anno - 2038) * 0.012)
            prod_petrolio_t[t] = prod_petrolio_t[t-1] + 0.08 * (prod_target - prod_petrolio_t[t-1]) + np.random.normal(0, 0.12)
        elif PAESE == "China":
            # Cina: produzione interna modesta (~4 Mb/g) e sostanzialmente piatta o in lieve declino.
            # La Cina è importatrice netta (~75% fabbisogno importato). Non c'è strategia "drill" domestica.
            # Il conflitto in ME non aumenta la produzione cinese, ma ne può aumentare il costo di approvvigionamento.
            prod_target_cn = PROFILO["prod_petrolio_2026"] * max(0.85, 1.0 - (anno - 2026) * 0.005)  # lieve declino fisiologico
            prod_petrolio_t[t] = prod_petrolio_t[t-1] + 0.05 * (prod_target_cn - prod_petrolio_t[t-1]) + np.random.normal(0, 0.05)
        else:
            # EU e altri: produzione marginale, lieve declino
            prod_target_eu = PROFILO["prod_petrolio_2026"] * max(0.5, 1.0 - (anno - 2026) * 0.015)
            prod_petrolio_t[t] = prod_petrolio_t[t-1] + 0.05 * (prod_target_eu - prod_petrolio_t[t-1]) + np.random.normal(0, 0.03)
        prod_petrolio_t[t] = max(0.1, prod_petrolio_t[t])
        
        # 3. ACCELERAZIONE EXPORT GNL VERSO L'EUROPA CON INERZIA DI COSTRUZIONE TERMINAL
        # La capacità massima è limitata fisicamente a 50 Gm3/anno, corrispondente al limite nominale 
        # nominale massimo di liquefazione dei terminali di export USA approvati e in costruzione entro il 2030.
        if anno < 2030:
            if intensita_run > 0:
                # Formula con protezione da divisione per zero (anno < 2030)
                target_gnl = export_gnl_t[t-1] + ((50.0 - export_gnl_t[t-1]) / (2030 - anno)) * intensita_run
            else:
                target_gnl = export_gnl_t[t-1] + 1.5
            # Smorzamento per riflettere i tempi fisici di posa tubi e navi gassiere (damping = 0.55)
            export_gnl_t[t] = export_gnl_t[t-1] + 0.55 * (target_gnl - export_gnl_t[t-1]) + np.random.normal(0, 0.3)
        else:
            # Dal 2030 in poi: plateau a 50 Gm3 (limite fisico di export) con smorzamento stocastico
            target_gnl = 50.0 + np.random.normal(0, 0.5)
            export_gnl_t[t] = export_gnl_t[t-1] + 0.4 * (target_gnl - export_gnl_t[t-1])
        export_gnl_t[t] = max(0.0, export_gnl_t[t])
        
        # 4. ATTIVAZIONE EMERGENZA DPA PER IL NUCLEARE (Soglia basata sull'intensità di questa traiettoria)
        dpa_active = (intensita_run >= 0.5)
        
        # --- VELOCITÀ PROGRAMMA URANIO USA (campionata per run) ---
        # Modella l'incertezza nel programma domestico HALEU:
        # 1.0 = nei tempi DoE, 0.6 = ritardi normativi NRC/budget
        _uranium_speed_usa = float(np.random.beta(4, 2))  # media ~0.67, range [0,1]

        # --- VANTAGGIO COSTRUTTIVO CINESE (Fonte: PDF credito libero informazioni) ---
        # La Cina finanzia il 71.2% del nucleare con fondi statali (vs ~6% USA, che è 94% privato).
        # Tempi di costruzione: 5-7 anni in Cina vs decenni in Occidente.
        # LCOE nucleare: ~2800$/kW in Cina vs ~9000$/kW USA, ~6600$/kW EU.
        # Filiera integrata internamente: nessun fornitore estero → no colli di bottiglia.
        # In Cina il "dpa_active" è quindi sempre True: lo Stato non dipende da emergenze
        # di mercato per costruire reattori; la pianificazione è continua e guidata dal governo.
        if PAESE == "China":
            dpa_active = True  # Lo Stato cinese guida la costruzione nucleare indipendentemente dal mercato
            # Nota: dpa_active=True per la Cina riflette la certezza della DIREZIONE strategica statale,
            # ma la variabilità stocastica delle capacità SMR e large reactor rimane nei rumori np.random
            # già presenti nelle formule (smr_cap + normal, large_reactor_growth con incertezza di politica).
            # Per mantenere un CI realistico, aggiungiamo un fattore di ritardo costruttivo stocastico:
            _ritardo_costruttivo = np.random.beta(5, 2)  # Beta(5,2): media ~0.71, range [0,1]
            # _ritardo_costruttivo vicino a 1 = costruzione nei tempi; < 1 = rallentamenti parziali

        # --- CAPACITÀ SMR (GW installati cumulativi) ---
        # Modello logistico diretto: SMR raggiungono un tetto fisico realistico di ~100 GW (USA, scenario DPA).
        # La curva logistica satura naturalmente; nessuna addizione cumulativa a ogni passo.
        # Flesso centrato nel 2035 (metà rampa di costruzione post-2028 con lead-time di 7 anni).
        if anno >= 2028:
            if dpa_active:
                if PAESE == "China":
                    # Cina: vantaggio 10-15 anni sugli USA negli SMR (Linglong One ACP100, HTR-PM).
                    # Cap SMR cinese: ~130 GW. Flesso anticipato al 2032.
                    # _ritardo_costruttivo (Beta-distribuito) allarga il CI: alcune traiettorie
                    # avanzano più lentamente per ritardi burocratici/tecnici inattesi.
                    _cn_ritardo = _ritardo_costruttivo  # già campionato sopra nel run MC
                    smr_cap = (130.0 * _cn_ritardo) + np.random.normal(0, 5.0)
                    smr_capacity_t[t] = smr_cap / (1.0 + np.exp(-0.40 * (anno - 2032)))
                else:
                    # USA/EU: Cap SMR ~100 GW (obiettivo ambizioso ma fisicamente plausibile entro il 2050)
                    smr_cap = 100.0 + np.random.normal(0, 2.0)
                    smr_capacity_t[t] = smr_cap / (1.0 + np.exp(-0.35 * (anno - 2035)))
            else:
                # Senza DPA: crescita molto più lenta, tetto a ~30 GW
                smr_cap_slow = 30.0
                smr_capacity_t[t] = smr_cap_slow / (1.0 + np.exp(-0.25 * (anno - 2038)))
        else:
            smr_capacity_t[t] = 0.0

        # --- CAPACITÀ LARGE REACTOR (GW, crescita lineare smorzata) ---
        if dpa_active:
            if PAESE == "China":
                _cn_cap_lr = 200.0 * _ritardo_costruttivo
                large_reactor_growth = float(np.clip(_cn_cap_lr / (1.0 + np.exp(-0.28 * (anno - 2034))) + np.random.normal(0, 1.5), 0.0, _cn_cap_lr))
            else:
                # USA: curva S inflection 2031, lead time NRC 8-12 anni
                large_reactor_growth = float(np.clip(120.0 / (1.0 + np.exp(-0.22 * (anno - 2031))) + np.random.normal(0, 1.0), 0.0, 120.0))
        else:
            large_reactor_growth = min((anno - 2026) * 1.0, 40.0)
            
        # Capacità totale: base + large reactor growth + SMR
        capacita_nucleare_t[t] = PROFILO["capacita_nucleare_2026"] + large_reactor_growth + smr_capacity_t[t]
        # Il cap rimane come hard ceiling dello scenario ottimistico (400 GW per entrambi USA e Cina)
        capacita_nucleare_t[t] = min(PROFILO["obiettivo_nucleare_2050"], capacita_nucleare_t[t])

        # --- BLACK SWAN: incidente nucleare -> moratoria -> meno capacita' ---
        _bs_nuc_rate = bse.get_delta("incidente_nucleare", anno).get("delta_nuc_cap_rate", 0.0)
        _bs_nuc_rate += bse.get_delta("shock_geopolitico_extra", anno).get("delta_nuc_cap_rate", 0.0)
        _bs_nuc_rate += bse.get_delta("crisi_finanziaria", anno).get("delta_nuc_cap_rate", 0.0)  # FEEDBACK: crisi -> nucleare
        if _bs_nuc_rate < 0 and capacita_nucleare_t[t] > PROFILO["capacita_nucleare_2026"]:
            # Il tasso di crescita viene moltiplicato: se rate = -0.6, la crescita aggiuntiva
            # rispetto alla base viene dimezzata + 40%
            _crescita_att = capacita_nucleare_t[t] - PROFILO["capacita_nucleare_2026"]
            _crescita_att *= max(0.1, 1.0 + _bs_nuc_rate)
            capacita_nucleare_t[t] = PROFILO["capacita_nucleare_2026"] + max(0.0, _crescita_att)
        
        # --- QUOTA % NUCLEARE NEL MIX ---
        # Guidata dalla capacità installata totale con elasticità al fattore di carico.
        # Usa la capacità *non cappata* come driver del ratio per evitare salite premature.
        # La Cina punta a 400 GW nucleari al 2050 (stesso target USA, fonte: PDF credito libero informazioni).
        # La quota % nel mix sarà però diversa per l'enorme domanda totale cinese.
        # USA: 400 GW su domanda ~6000 TWh → ~28% mix. Cina: 400 GW su domanda ~15000 TWh → ~18% mix.
        # target_nuc_2050: quota % nucleare nel mix al 2050.
        # USA: il nucleare cresce da 100 a ~315 GW (+215 GW) ma la domanda totale cresce
        # anch'essa per AI e elettrificazione. La quota % resta moderata: 18% (da ~7% attuale).
        # 28% era troppo aggressivo e sottraeva quota al fossile annullando la fossil_compensation.
        # Cina: 18% (da ~5% attuale, obiettivo ambizioso ma domanda cresce enormemente)
        # EU: 22% (mix già nucleare-intensivo, Francia ancora dominante)
        target_nuc_2050 = 18.0 if PAESE == "United States" else (18.0 if PAESE == "China" else 22.0)
        # Cap obiettivo non cappato: per la Cina è molto più alto (400 GW totali = +345 GW da base 55 GW)
        if PAESE == "China":
            cap_obiettivo_non_cappato = PROFILO["capacita_nucleare_2026"] + (200.0 if dpa_active else 80.0) + (145.0 if dpa_active else 50.0)
        else:
            cap_obiettivo_non_cappato = PROFILO["capacita_nucleare_2026"] + (120.0 if dpa_active else 40.0) + (100.0 if dpa_active else 30.0)
        nuclear_ratio = np.clip(
            (capacita_nucleare_t[t] - PROFILO["capacita_nucleare_2026"]) /
            max(1.0, cap_obiettivo_non_cappato - PROFILO["capacita_nucleare_2026"]),
            0.0, 1.0
        )
        quota_nucleare_t[t] = quota_nucleare_storica[-1] + nuclear_ratio * (target_nuc_2050 - quota_nucleare_storica[-1]) + np.random.normal(0, 0.2)
        quota_nucleare_t[t] = np.clip(quota_nucleare_t[t], 0.0, 100.0)
        
        # ABBATTIMENTO DIPENDENZA URANIO — modello per paese con lead time fisico realistico
        #
        # CALIBRAZIONE FISICAMENTE MOTIVATA (fonte: IAEA, DoE, NRC):
        # ─────────────────────────────────────────────────────────────
        # USA: partono al 95%. Il programma DPA "Nuclear Dominance 3 by 33" significa
        #      avvio costruzione impianti HALEU entro 2033, NON produzione a pieno regime.
        #      Lead time reale: 10-15 anni per arricchimento (Centrus Ohio: primo lotto 2023,
        #      scala industriale ~2035-2038). Floor realistico: 40-50% al 2033, 20-25% al 2050.
        #      La "Prohibiting Russian Uranium Imports Act" riduce dipendenza dalla Russia
        #      (~50% storicamente) ma NON elimina dipendenza da Canada/Kazakhstan.
        #
        # China: partono al 70%. Filiera nucleare integrata con capacità di arricchimento
        #        domestica in espansione (CNNC). Floor raggiungibile: ~15% al 2040.
        #        Lead time più corto per capacità costruttiva statale già attiva.
        #
        # EU: partono al 90%. Dipendenza strutturale da Russia/Kazakhstan difficile da ridurre.
        #     Floor realistico: ~50-60% al 2050 (Urenco copre ~30% del fabbisogno EU).

        if PAESE == "United States":
            uranium_floor = 40.0   # floor realistico 2050: anche con DPA+Centrus+Canada,
            # la dipendenza non scende sotto 40% perché Canada da solo copre ~25% e
            # il programma HALEU domestico non può scalare abbastanza per 400 GW al 2050.
            # La "Prohibiting Russian Uranium Imports Act" rimuove Russia (~50% storico)
            # ma non elimina dipendenza da Kazakhstan/Niger/Australia.
            if anno <= 2033:
                _calo_fase1 = 5.0 * (anno - 2026) / 7.0
                _noise_ura = np.random.normal(0, 0.8)
                dipendenza_uranio_t[t] = max(
                    PROFILO["dipendenza_uranio_2026"] - _calo_fase1 + _noise_ura,
                    85.0
                )
            elif anno <= 2041:
                _calo_fase2 = 1.5 * _uranium_speed_usa  # ridotto da 3.0 a 1.5 pp/anno
                target_ura = dipendenza_uranio_t[t-1] - _calo_fase2
                _noise_ura = np.random.normal(0, 1.0)
                dipendenza_uranio_t[t] = max(55.0, target_ura + _noise_ura)
            else:
                target_ura = uranium_floor + (dipendenza_uranio_t[t-1] - uranium_floor) * 0.88
                _noise_ura = np.random.normal(0, 0.5)
                dipendenza_uranio_t[t] = max(uranium_floor, target_ura + _noise_ura)

        elif PAESE == "China":
            # Floor 42%: dipendenza strutturale da Kazakhstan/Russia per arricchimento.
            # La domanda assoluta triplica con i 400 GW — autonomia piena entro 2050 impossibile.
            uranium_floor = 42.0
            _uranium_speed = _ritardo_costruttivo
            if anno <= 2033:
                _calo = 6.0 * (anno - 2026) / 7.0 * _uranium_speed
                dipendenza_uranio_t[t] = max(60.0, PROFILO["dipendenza_uranio_2026"] - _calo + np.random.normal(0, 1.2))
            elif anno <= 2043:
                target_dip = uranium_floor + (dipendenza_uranio_t[t-1] - uranium_floor) * (1.0 - 0.04 * _uranium_speed)
                dipendenza_uranio_t[t] = max(uranium_floor, dipendenza_uranio_t[t-1] + 0.12 * (target_dip - dipendenza_uranio_t[t-1]) + np.random.normal(0, 1.0))
            else:
                target_dip = uranium_floor + (dipendenza_uranio_t[t-1] - uranium_floor) * 0.97
                dipendenza_uranio_t[t] = max(uranium_floor, dipendenza_uranio_t[t-1] + 0.10 * (target_dip - dipendenza_uranio_t[t-1]) + np.random.normal(0, 0.6))

        else:  # EU
            uranium_floor = 45.0   # Urenco ~30% fabbisogno, diversificazione parziale
            target_dip = uranium_floor + (dipendenza_uranio_t[t-1] - uranium_floor) * 0.97  # lentissima riduzione
            _noise_ura = np.random.normal(0, 0.5)
            dipendenza_uranio_t[t] = max(uranium_floor, dipendenza_uranio_t[t-1] + 0.2*(target_dip - dipendenza_uranio_t[t-1]) + _noise_ura)

        # 5. EVOLUZIONE RINNOVABILI (Crescita con inerzia strutturale)
        standard_rin_growth = base_quota_rinnovabile_logistica[t] - base_quota_rinnovabile_logistica[t-1]

        # --- BLACK SWAN: shock su tasso di crescita rinnovabili ---
        _bs_rin_rate  = bse.get_delta("crisi_finanziaria",  anno).get("delta_rin_rate", 0.0)
        _bs_rin_rate += bse.get_delta("tech_breakdown",     anno).get("delta_rin_rate", 0.0)
        _bs_rin_rate += bse.get_delta("tech_breakthrough",  anno).get("delta_rin_rate", 0.0)
        # delta_rin_rate e' una variazione % annua applicata al tasso di crescita
        _rin_growth_adj = standard_rin_growth + _bs_rin_rate * 0.1  # scala: 0.1 TWh per % delta
        # FRENO TRUMP (USA 2026-2032): -0.40pp/anno in 2026, zero in 2032
        if PAESE == "United States" and anno <= 2032:
            _rin_growth_adj += -0.40 * (1.0 - (anno - 2026) / 6.0)

        # BLACK SWAN: breakthrough alza anche il tetto max_rin per questa traiettoria
        _bs_max_rin  = bse.get_delta("tech_breakdown",    anno).get("delta_max_rin", 0.0)
        _bs_max_rin += bse.get_delta("tech_breakthrough", anno).get("delta_max_rin", 0.0)
        _max_rin_run = np.clip(max_rin + _bs_max_rin, 30.0, 100.0)

        target_rin = quota_rinnovabile_t[t-1] + _rin_growth_adj + np.random.normal(0, 0.5)
        quota_rinnovabile_t[t] = quota_rinnovabile_t[t-1] + 0.65 * (target_rin - quota_rinnovabile_t[t-1])
        quota_rinnovabile_t[t] = np.clip(quota_rinnovabile_t[t], 0.0, _max_rin_run)
        
        # 6. RICONCILIAZIONE MIX ENERGETICO (Somma = 100%)
        # PROBLEMA ORIGINALE: usare il fossile come "swing producer" residuale significava che
        # tutta la crescita del nucleare (+19pp in scenario DPA) veniva sottratta al fossile,
        # rendendo nulla la fossil_compensation. Nel documento USA il nucleare cresce per
        # soddisfare la domanda AGGIUNTIVA di AI/semiconduttori, non per rimpiazzare il gas.
        # SOLUZIONE: il mismatch viene distribuito proporzionalmente tra le tre fonti,
        # preservando le quote relative e rispettando la fossil_compensation calcolata sopra.
        _tot_raw = quota_fossile_t[t] + quota_nucleare_t[t] + quota_rinnovabile_t[t]
        if _tot_raw > 0:
            quota_fossile_t[t]    = (quota_fossile_t[t]    / _tot_raw) * 100.0
            quota_nucleare_t[t]   = (quota_nucleare_t[t]   / _tot_raw) * 100.0
            quota_rinnovabile_t[t] = (quota_rinnovabile_t[t] / _tot_raw) * 100.0

    # Salvataggio traiettorie nello storico Monte Carlo
    mc_quota_fossile[i] = quota_fossile_t
    mc_quota_nucleare[i] = quota_nucleare_t
    mc_quota_rinnovabile[i] = quota_rinnovabile_t
    mc_prod_petrolio[i] = prod_petrolio_t
    mc_export_gnl[i] = export_gnl_t
    mc_capacita_nucleare[i] = capacita_nucleare_t
    mc_smr_capacity[i] = smr_capacity_t
    mc_dipendenza_uranio[i] = dipendenza_uranio_t
    
    # 7. CALCOLO VARIABILI AGGREGATE POST-LOOP
    sqf = quota_fossile_t
    
    # Più risparmio fossile → MAGGIORE efficienza (correlazione positiva)
    risparmio_fossile_perc = base_quota_fossile_logistica - sqf
    
    # Formula moltiplicativa dell'efficienza (Anomaly 1.1):
    # base_efficienza migliora nel tempo per progresso tecnologico (~0.8% all'anno)
    base_efficienza = np.exp(-0.008 * (anni_futuri - 2026))
    # Risparmio fossile (positivo = meno fossili della baseline → più efficiente)
    # Eccesso fossile (negativo) penalizza leggermente l'efficienza (sistema più inquinato e meno ottimizzato)
    risparmio_clip = np.clip(risparmio_fossile_perc, -50, 100)
    fattore_efficienza = base_efficienza * (1.0 - 0.3 * (risparmio_clip / 100.0))
    fattore_efficienza = np.clip(fattore_efficienza, 0.65, 1.05)  # leggero peggioramento possibile (>1) se fossili in eccesso
    
    # Calcolo dei consumi totali (TWh): separiamo il consumo generale (scalato da fattore_efficienza)
    # dall'impatto dei data center IA che ha già la sua efficienza dedicata ia_efficienza (evita double counting - Anomaly 2.2)
    # COMMENTI UNITA': s_cons è espresso in TWh (unità fisica del dataset OWID)
    s_cons = (base_consumo_totale + feedback_climatico_twh) * fattore_efficienza + ia_consumo
    mc_consumo[i] = s_cons
    
    if PAESE == 'China':
        mc_intensita[i] = np.clip(base_intensita * fattore_efficienza, PROFILO["ss_efficiency_floor"], 5)
    else:
        mc_intensita[i] = np.clip(base_intensita * fattore_efficienza, 0.28, 5)

    # Accoppiamento Geopolitico Reale: dipendenza legata al consumo fossile simulato vs baseline.
    # FIX DISCONTINUITÀ: calcoliamo prima la traiettoria grezza, poi sottraiamo il valore al t=0
    # (che include il contributo di nuc_dipendenza e rumore al primo passo) in modo che la
    # proiezione parta esattamente da dipendenza_storica[-1] nel grafico.
    _dip_raw = base_dipendenza + (sqf - base_quota_fossile_logistica) * 0.4 + nuc_dipendenza + rumore_mc_dip
    _dip_offset_t0 = _dip_raw[0] - dipendenza_storica[-1]
    mc_dipendenza[i] = _dip_raw - _dip_offset_t0
    
    crescita_dip_verde = risparmio_fossile_perc * 0.5
    _dv_raw = base_dip_verde + crescita_dip_verde + miglioramento_supply_chain + rumore_mc_dip
    _dv_offset_t0 = _dv_raw[0] - dip_verde_storica[-1]
    mc_dip_verde[i] = _dv_raw - _dv_offset_t0
    
    mc_consumo_fossile_twh[i] = s_cons * (sqf / 100.0)
    mc_consumo_nucleare_twh[i] = s_cons * (quota_nucleare_t / 100.0)
    mc_consumo_rinnovabile_twh[i] = s_cons * (quota_rinnovabile_t / 100.0)
    mc_energia_per_capita[i] = (s_cons * 1e6) / popolazione_futura
    
    # 1. Elettrificazione accelera quando aumentano le rinnovabili
    quota_rin = 100.0 - sqf  # quota rinnovabili + nucleare
    boost_elettrico = np.clip((quota_rin - 40) * 0.4, 0, 25)  # +25% max
    elettrificazione_con_boost = elettrificazione_base + boost_elettrico
    mc_elettrificazione[i] = np.clip(elettrificazione_con_boost + np.random.normal(0, 1.0, len(anni_futuri)), 0, 100)
    
    moltiplicatore_capex_locale = moltiplicatore_capex.copy()

    # Shock Tecnologici e Geopolitici addizionali con calcolo incrementale
    if PAESE == 'United States':
        shock_tanbreez = calcola_shock_incrementale(anni_futuri, PROFILO["tanbreez_anno"], PROFILO["tanbreez_impatto_export"], k=0.8, flesso=3)
        mc_dip_verde[i] += shock_tanbreez
        moltiplicatore_capex_locale[anni_futuri >= 2029] *= 0.90
    elif PAESE == 'China':
        # --- SHOCK TECNOLOGICI E STRATEGICI CINA (Fonte: PDF credito libero informazioni) ---

        # 1. BATTERIE SOLID-STATE: accelerazione elettrificazione dal 2026
        boost_elec = calcola_shock_incrementale(anni_futuri, PROFILO["ss_battery_anno"], PROFILO["ss_elec_boost"], k=0.8, flesso=3)
        mc_elettrificazione[i] += boost_elec

        # 2. PARADOSSO DEL CARBONE: la Cina è il primo emettitore CO2 mondiale nonostante i record
        #    nelle rinnovabili. 97 GW carbone in costruzione + 152 GW in progetto.
        #    Il carbone (~60% del mix attuale) non sparirà prima del 2035+.
        #    Post-2030 però: picco emissioni, poi stretta normativa → decrescita accelerata del carbone.
        anni_post_coal_peak = np.where(anni_futuri > PROFILO["carbone_peak_anno"],
                                       anni_futuri - PROFILO["carbone_peak_anno"], 0)
        coal_exit_shock = -anni_post_coal_peak * PROFILO["carbone_exit_rate"] * 8.0  # -8 TWh/anno post-peak
        mc_consumo_fossile_twh[i] = mc_consumo_fossile_twh[i] + coal_exit_shock * (mc_consumo[i] / 10000.0)

        # 3. DOMINANZA SOLARE E HVDC: la Cina produce 81% celle FV globali, 35% energia solare mondiale.
        #    Le reti HVDC (High Voltage Direct Current) trasportano energia da deserto Gobi→coste
        #    con solo 3% perdite ogni 1000 km. Questo abbatte il disallineamento geografico prod/consumo.
        #    Effetto: riduzione del "curtailment" → più rinnovabili integrate → meno fossili.
        anni_hvdc = np.clip(anni_futuri - 2028, 0, None)  # HVDC a regime dal 2028
        hvdc_efficienza_boost = np.minimum(anni_hvdc * 0.3, 6.0)  # max +6% rinnovabili integrate
        mc_quota_rinnovabile[i] = np.clip(mc_quota_rinnovabile[i] + hvdc_efficienza_boost, 0, 100)

        # 4. VANTAGGIO CAPEX NUCLEARE: costo di costruzione ~2800$/kW (vs 9000$/kW USA).
        #    Moltiplicatore CAPEX ridotto strutturalmente del fattore lcoe_vantaggio_factor.
        moltiplicatore_capex_locale *= PROFILO["lcoe_vantaggio_factor"]
    elif PAESE == 'European Union (27)':
        bonus_indip = calcola_shock_incrementale(anni_futuri, PROFILO["plastic_h2_anno"], PROFILO["plastic_indip_bonus"], k=0.8, flesso=3)
        mc_dipendenza[i] += bonus_indip
        moltiplicatore_capex_locale[anni_futuri >= 2027] *= (1.0 - PROFILO["plastic_capex_discount"])

    # 2. Storage Stabilità Rete (np.where per vettori numpy)
    storage_perc = np.where(quota_rin <= 40, quota_rin * 0.5, 20 + (quota_rin - 40) * 0.625)
    storage_perc = np.clip(storage_perc, 0, 60)
    mc_storage[i] = s_cons * (storage_perc / 100.0)
    
    # 3. Emissioni CO2 (Mt) basate su fattore specifico del paese
    mc_emissioni[i] = mc_consumo_fossile_twh[i] * PROFILO["emission_factor"]
    mc_emissioni_cum[i] = np.sum(mc_emissioni[i])
    
    # 4. CAPEX su PIL
    # Include crescita rinnovabili E nucleare: nel documento USA la spesa principale è
    # il programma nucleare DPA (SMR + grandi reattori). Costo stimato: ~6000$/kW nucleare,
    # ~1200$/kW rinnovabili. Convertiamo TWh in GW con fattore di carico medio.
    rinnovabili_diff = np.diff(mc_consumo_rinnovabile_twh[i], prepend=storico_rinnovabile_twh[-1])

    # --- BLACK SWAN: crisi finanziaria comprime CAPEX disponibile ---
    _bs_capex_arr = np.array([
        1.0 + bse.get_delta("crisi_finanziaria", a).get("delta_capex_mult", 0.0)
        for a in anni_futuri
    ])
    moltiplicatore_capex_locale *= np.clip(_bs_capex_arr, 0.3, 2.0)

    # 4. CAPEX su PIL — calcolato su capacità GW installata, non su TWh derivati da quote %
    # Le quote % dipendono dalla normalizzazione e possono essere distorte.
    # La capacità GW è il dato fisico reale del piano DPA: ~6000$/kW nucleare, ~1200$/kW rinnovabili.
    # Conversione: 1 GW nucleare (fattore carico 90%) = ~7.9 TWh/anno → costo ~6 mld$/GW
    # Conversione: 1 GW rinnovabili (fattore carico 30%) = ~2.6 TWh/anno → costo ~1.2 mld$/GW
    cap_nuc_gw  = mc_capacita_nucleare[i]   # GW totali installati
    cap_nuc_diff = np.diff(cap_nuc_gw, prepend=PROFILO["capacita_nucleare_2026"])
    cap_nuc_diff = np.maximum(0.0, cap_nuc_diff)

    rinnovabili_diff = np.diff(mc_consumo_rinnovabile_twh[i], prepend=storico_rinnovabile_twh[-1])

    # --- BLACK SWAN: crisi finanziaria comprime CAPEX disponibile ---
    _bs_capex_arr = np.array([
        1.0 + bse.get_delta("crisi_finanziaria", a).get("delta_capex_mult", 0.0)
        for a in anni_futuri
    ])
    moltiplicatore_capex_locale *= np.clip(_bs_capex_arr, 0.3, 2.0)

    # CAPEX nucleare: 6 mld$/GW; CAPEX rinnovabili: 1.0 mld$/TWh
    capex_nuc = cap_nuc_diff * 6.0 * moltiplicatore_capex_locale
    capex_rin = np.maximum(0.0, rinnovabili_diff) * 1.0 * moltiplicatore_capex_locale
    capex_mld = capex_rin + capex_nuc
    # pil_run è il PIL stocastico di questo run (perturbato da crisi finanziarie BSE).
    # Quando c'è una crisi: CAPEX cala (delta_capex_mult) E il PIL cala (delta_pil)
    # → il rapporto CAPEX/PIL può andare in entrambe le direzioni a seconda della magnitudo.
    # Questo è coerente con la realtà: in alcune crisi il ratio sale (PIL crolla più del CAPEX),
    # in altre scende (gli investimenti vengono tagliati più dell'economia).
    mc_capex_pil[i] = (capex_mld / pil_run) * 100.0
    mc_pil[i]       = pil_run

# ==========================================
# 6. CALCOLO STATISTICHE E MEDIANE DI SCENARIO
# Aggiorna pil_futuro_mld alla mediana stocastica (usato nei grafici CAPEX)
pil_futuro_mld = np.median(mc_pil, axis=0)
pil_mld_low, pil_mld_high = np.percentile(mc_pil, [2.5, 97.5], axis=0)
# ==========================================
scenario_quota_fossile = np.median(mc_quota_fossile, axis=0)
sqf_low, sqf_high = np.percentile(mc_quota_fossile, [2.5, 97.5], axis=0)

scenario_quota_nucleare = np.median(mc_quota_nucleare, axis=0)
sqn_low, sqn_high = np.percentile(mc_quota_nucleare, [2.5, 97.5], axis=0)

scenario_quota_rinnovabile = np.median(mc_quota_rinnovabile, axis=0)
sqr_low, sqr_high = np.percentile(mc_quota_rinnovabile, [2.5, 97.5], axis=0)

scenario_consumo = np.median(mc_consumo, axis=0)
sc_low, sc_high = np.percentile(mc_consumo, [2.5, 97.5], axis=0)

scenario_intensita = np.median(mc_intensita, axis=0)
si_low, si_high = np.percentile(mc_intensita, [2.5, 97.5], axis=0)

scenario_dipendenza = np.median(mc_dipendenza, axis=0)
sd_low, sd_high = np.percentile(mc_dipendenza, [2.5, 97.5], axis=0)

scenario_dip_verde = np.median(mc_dip_verde, axis=0)
sdv_low, sdv_high = np.percentile(mc_dip_verde, [2.5, 97.5], axis=0)

consumo_fossile_twh = np.median(mc_consumo_fossile_twh, axis=0)
consumo_nucleare_twh = np.median(mc_consumo_nucleare_twh, axis=0)
consumo_rinnovabile_twh = np.median(mc_consumo_rinnovabile_twh, axis=0)

energia_per_capita_futura = np.median(mc_energia_per_capita, axis=0)
epc_low, epc_high = np.percentile(mc_energia_per_capita, [2.5, 97.5], axis=0)

scenario_elettrificazione = np.median(mc_elettrificazione, axis=0)
scenario_storage = np.median(mc_storage, axis=0)

scenario_emissioni = np.median(mc_emissioni, axis=0)
em_low, em_high = np.percentile(mc_emissioni, [2.5, 97.5], axis=0)
mediana_emissioni_cum = np.median(mc_emissioni_cum)

scenario_capex_pil = np.median(mc_capex_pil, axis=0)
capex_low, capex_high = np.percentile(mc_capex_pil, [2.5, 97.5], axis=0)

# Statistiche per nuove variabili
scenario_prod_petrolio = np.median(mc_prod_petrolio, axis=0)
pet_low, pet_high = np.percentile(mc_prod_petrolio, [2.5, 97.5], axis=0)

scenario_export_gnl = np.median(mc_export_gnl, axis=0)
gnl_low, gnl_high = np.percentile(mc_export_gnl, [2.5, 97.5], axis=0)

scenario_capacita_nuc = np.median(mc_capacita_nucleare, axis=0)
nuc_cap_low, nuc_cap_high = np.percentile(mc_capacita_nucleare, [2.5, 97.5], axis=0)

scenario_smr_capacity = np.median(mc_smr_capacity, axis=0)
smr_low, smr_high = np.percentile(mc_smr_capacity, [2.5, 97.5], axis=0)

scenario_uranio = np.median(mc_dipendenza_uranio, axis=0)
ura_low, ura_high = np.percentile(mc_dipendenza_uranio, [2.5, 97.5], axis=0)

err_std = np.std(mc_emissioni_cum) / np.sqrt(N_SIMULAZIONI)

# ===== DEBUG RUNTIME =====
print("\n=== DEBUG VALORI CHIAVE ===")
print(f"[D1] Baseline rin 2026={base_quota_rinnovabile_logistica[0]:.1f}% -> 2050={base_quota_rinnovabile_logistica[-1]:.1f}%")
print(f"[D2] Baseline fossile 2026={base_quota_fossile_logistica[0]:.1f}% -> 2050={base_quota_fossile_logistica[-1]:.1f}%")
print(f"[D3] Fossile MEDIANA  2026={scenario_quota_fossile[0]:.1f}% | 2030={scenario_quota_fossile[4]:.1f}% | 2040={scenario_quota_fossile[14]:.1f}% | 2050={scenario_quota_fossile[-1]:.1f}%")
print(f"[D4] Non-fossile MED  2026={100-scenario_quota_fossile[0]:.1f}% | 2050={100-scenario_quota_fossile[-1]:.1f}%")
print(f"[D5] Rin. MC mediana  2026={scenario_quota_rinnovabile[0]:.1f}% | 2050={scenario_quota_rinnovabile[-1]:.1f}%")
print(f"[D6] PAESE attivo = '{PAESE}'")
print("============================\n")

print(f"\n--- KEY TAKEAWAYS ({PAESE} - Scenario Guerra Medio Oriente a {intensita_conflitto_me:.2f}) ---")
print(f"Emissioni Cumulative (2026-2050): {mediana_emissioni_cum/1000:.1f} Gt CO2")
print(f"Errore standard emissioni cumulative: {err_std/1000:.3f} Gt CO2")
print(f"Sforzo Economico Massimo (CAPEX/PIL): {np.max(scenario_capex_pil):.2f}% del PIL annuo (Picco Bolla Speculativa Shiller nel 2027)")
if PAESE == "United States":
    print(f"Produzione Petrolifera USA Finale (2050): {scenario_prod_petrolio[-1]:.2f} Mb/g")
    print(f"Esportazioni GNL verso Europa (2030): {scenario_export_gnl[anni_futuri == 2030][0]:.1f} Gm3/anno")
print(f"Capacità Nucleare Installata Finale (2050): {scenario_capacita_nuc[-1]:.1f} GW (Target {PROFILO['obiettivo_nucleare_2050']:.0f} GW)")
print(f"Dipendenza Uranio Estero (2033): {scenario_uranio[anni_futuri == 2033][0]:.1f}%")
if PAESE == "China":
    print(f"[CINA] Obiettivo nucleare 2050: {PROFILO['obiettivo_nucleare_2050']:.0f} GW (Fonte: PDF credito libero informazioni)")
    print(f"[CINA] Capacità SMR mediana (2040): {np.median(mc_smr_capacity[:, anni_futuri == 2040]):.1f} GW (vantaggio 10-15 anni su USA)")
    print(f"[CINA] Dominanza solare globale: 81% celle FV prodotte, 35% energia solare mondiale")
    print(f"[CINA] Paradosso carbone: ~60% mix attuale, peak emissioni previsto pre-2030")
    print(f"[CINA] CFETR (fusione): costruzione dal 2030, operativo ~2040; 1.5 mld$/anno R&D fusione")
print("---------------------------------")

# --- BLACK SWAN ENGINE: statistiche aggregate su 1000 run ---
print(f"\n--- BLACK SWAN ENGINE (frequenze su {N_SIMULAZIONI} run) ---")
for cat, n_run in registro_bs.items():
    freq_pct = n_run / N_SIMULAZIONI * 100
    print(f"  {cat:30s}: {freq_pct:5.1f}% dei run ({n_run}/{N_SIMULAZIONI})")
print("---------------------------------")

# ==========================================
# 7. VISUALIZZAZIONE E DASHBOARD GRAFICHE
# ==========================================
# DASHBOARD 1: Analisi di Produzione ed Efficienza
fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(f"Analisi di Produzione ed Efficienza ({PAESE})", fontsize=16, fontweight='bold')

# Stackplot 3 componenti: Fossili, Nucleare, Rinnovabili
axs1[0, 0].stackplot(anni_storici, storico_fossile_twh, storico_nucleare_twh, storico_rinnovabile_twh, 
                     labels=['Fossili Storico', 'Nucleare Storico', 'Rinnovabili Storico'], 
                     colors=['#666666', '#FFB74D', '#4CAF50'], alpha=0.7)
axs1[0, 0].stackplot(anni_futuri, consumo_fossile_twh, consumo_nucleare_twh, consumo_rinnovabile_twh, 
                     labels=['Fossili Scenario', 'Nucleare Scenario', 'Rinnovabili Scenario'], 
                     colors=['#333333', '#FF9800', '#81C784'], alpha=0.9)
axs1[0, 0].plot(anni_futuri, scenario_consumo, color='white', lw=1, ls='-', label="Totale Scenario")
axs1[0, 0].fill_between(anni_futuri, sc_low, sc_high, color='black', alpha=0.15)
axs1[0, 0].plot(anni_futuri, base_consumo_totale, 'gray', ls='--', lw=2, label="Baseline Senza Shock")
axs1[0, 0].axvline(x=2026, color='red', linestyle=':', alpha=0.7, label='Inizio Conflitto M.O.')
axs1[0, 0].set_title("Produzione/Consumo Totale (TWh)")
axs1[0, 0].set_ylabel("TWh")
axs1[0, 0].grid(alpha=0.3)
axs1[0, 0].legend(loc='upper left', fontsize=8)

# Quota Fossili (%)
axs1[0, 1].plot(anni_storici, quota_fossile, 'k-', lw=2, label="Storico")
axs1[0, 1].plot(anni_futuri, base_quota_fossile_logistica, 'gray', ls='--', lw=2, label="Baseline")
axs1[0, 1].plot(anni_futuri, scenario_quota_fossile, 'purple', lw=3, label="Scenario Mediano (Shock)")
axs1[0, 1].fill_between(anni_futuri, sqf_low, sqf_high, color='purple', alpha=0.2)
axs1[0, 1].axvline(x=2026, color='red', linestyle=':', alpha=0.7, label='Inizio Conflitto M.O.')
axs1[0, 1].set_title("Quota Fossili nel Mix (%)")
axs1[0, 1].set_ylabel("%")
axs1[0, 1].set_ylim(0, 100)
axs1[0, 1].grid(alpha=0.3)
axs1[0, 1].legend(fontsize=8)

# Intensità Energetica
axs1[1, 0].plot(anni_storici, intensita_storica, 'k-', lw=2, label="Storico")
axs1[1, 0].plot(anni_futuri, base_intensita, 'gray', ls='--', lw=2, label="Baseline")
axs1[1, 0].plot(anni_futuri, scenario_intensita, 'purple', lw=3, label="Scenario Mediano")
axs1[1, 0].fill_between(anni_futuri, si_low, si_high, color='purple', alpha=0.2)
axs1[1, 0].axvline(x=2026, color='red', linestyle=':', alpha=0.7, label='Inizio Conflitto M.O.')
axs1[1, 0].set_title("Intensità Energetica (Meno è Meglio)")
axs1[1, 0].grid(alpha=0.3)
axs1[1, 0].legend(fontsize=8)

# Energia per Capita
axs1[1, 1].plot(anni_storici, energia_per_capita_storica, 'b-', lw=2, label="Storico (MWh/ab)")
axs1[1, 1].plot(anni_futuri, base_per_capita, 'gray', ls='--', lw=2, label="Baseline")
axs1[1, 1].plot(anni_futuri, energia_per_capita_futura, 'dodgerblue', lw=3, label="Scenario Mediano (MWh/ab)")
axs1[1, 1].fill_between(anni_futuri, epc_low, epc_high, color='dodgerblue', alpha=0.2)
axs1[1, 1].axvline(x=2026, color='red', linestyle=':', alpha=0.7, label='Inizio Conflitto M.O.')
axs1[1, 1].set_title("Energia PRIMARIA per Abitante")
axs1[1, 1].set_ylabel("MWh / persona")
axs1[1, 1].grid(alpha=0.3)
axs1[1, 1].legend(fontsize=8)
fig1.tight_layout()

# DASHBOARD 2: Geopolitica e Adozione Tecnologica (2x2 speculare per mostrare le nuove equazioni)
fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(f"Geopolitica, Produzione e Adozione Tecnologica ({PAESE})", fontsize=16, fontweight='bold')

# Quota Rinnovabili e Nucleare (%)
axs2[0, 0].plot(anni_storici, quota_rinnovabile + quota_nucleare_storica, 'green', lw=2, label="Storico (Rin+Nuc)")
axs2[0, 0].plot(anni_futuri, base_quota_rinnovabile_logistica + base_quota_nucleare, 'lightgreen', ls='--', lw=2, label="Baseline (Rin+Nuc)")
axs2[0, 0].plot(anni_futuri, 100 - scenario_quota_fossile, 'forestgreen', lw=3, label="Scenario Verde (Rin+Nuc)")
axs2[0, 0].fill_between(anni_futuri, 100-sqf_high, 100-sqf_low, color='forestgreen', alpha=0.2)
axs2[0, 0].axvline(x=2026, color='red', linestyle=':', alpha=0.7, label='Inizio Conflitto M.O.')
axs2[0, 0].set_title("Quota Non-Fossile nel Mix (%)")
axs2[0, 0].set_ylim(0, 100)
axs2[0, 0].grid(alpha=0.3)
axs2[0, 0].legend(fontsize=8)

# Dipendenze Geopolitiche classiche
axs2[0, 1].plot(anni_storici, dipendenza_storica, 'k-', lw=2)
axs2[0, 1].plot(anni_futuri, scenario_dipendenza, 'darkorange', lw=3, label="Dipendenza Fossile (Mediana)")
axs2[0, 1].fill_between(anni_futuri, sd_low, sd_high, color='darkorange', alpha=0.2)
axs2[0, 1].plot(anni_storici, dip_verde_storica, 'green', lw=2, linestyle=':')
axs2[0, 1].plot(anni_futuri, scenario_dip_verde, 'green', lw=3, label="Dipendenza Verde (Minerali)")
axs2[0, 1].fill_between(anni_futuri, sdv_low, sdv_high, color='green', alpha=0.2)
axs2[0, 1].axhline(0, color='red', linestyle='--', lw=1) 
axs2[0, 1].axvline(x=2026, color='red', linestyle=':', alpha=0.7, label='Inizio Conflitto M.O.')
_dip_data_label = "(net import % da OWID)" if _prod_tot is not None else "(proxy sintetico)"
axs2[0, 1].set_title(f"Geopolitica: Dipendenze Storiche {_dip_data_label}")
_ylabel_dip = "Net Energy Import %" if len(_serie_pil_wb) >= 10 or True else "% Importazioni (proxy)"
axs2[0, 1].set_ylabel("Net Energy Import %  (positivo = dipendente, negativo = esportatore)")
axs2[0, 1].grid(alpha=0.3)
axs2[0, 1].legend(fontsize=8)

# NUOVO SUBPLOT 1, 0: "Drill, Baby, Drill" e Export GNL USA (Guerra Medio Oriente)
color_pet = 'tab:blue'
title_petrolio = "Produzione Petrolio & Export GNL USA" if PAESE == "United States" else f"Produzione Petrolio & Indipendenza Energetica ({PAESE})"
axs2[1, 0].set_title(f"Impatto Guerra: {title_petrolio}", fontsize=11, weight='bold')
axs2[1, 0].set_xlabel("Anno")
axs2[1, 0].set_ylabel("Produzione Petrolio Greggio (Mb/g)", color=color_pet)
_label_prod_pet = f"Produzione Petrolio {PAESE} (Mb/g)"
axs2[1, 0].plot(anni_futuri, scenario_prod_petrolio, color=color_pet, lw=3, label=_label_prod_pet)
axs2[1, 0].fill_between(anni_futuri, pet_low, pet_high, color=color_pet, alpha=0.2)
axs2[1, 0].tick_params(axis='y', labelcolor=color_pet)
axs2[1, 0].grid(alpha=0.3)

ax2_twin = axs2[1, 0].twinx()
color_gnl = 'tab:red'
ax2_twin.set_ylabel("Export GNL verso Europa (Gm3/anno)", color=color_gnl)
_label_gnl = f"Export GNL {PAESE} (Gm3/anno)"
ax2_twin.plot(anni_futuri, scenario_export_gnl, color=color_gnl, lw=3, ls='--', label=_label_gnl)
ax2_twin.fill_between(anni_futuri, gnl_low, gnl_high, color=color_gnl, alpha=0.2)
_label_target_gnl = 'Target GNL 2030 (50 Gm3)' if PAESE == 'United States' else f'Livello GNL corrente ({PAESE})'
ax2_twin.axhline(50.0 if PAESE == 'United States' else scenario_export_gnl[-1],
                 color='gray', linestyle=':', label=_label_target_gnl)
ax2_twin.tick_params(axis='y', labelcolor=color_gnl)

lines_1, labels_1 = axs2[1, 0].get_legend_handles_labels()
lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
axs2[1, 0].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=8)

# NUOVO SUBPLOT 1, 1: DPA Nucleare (Nuclear Dominance & SMR)
color_ura = 'tab:orange'
title_nuc = "Emergenza DPA: Capacità Nucleare & Indipendenza Uranio" if PAESE == "United States" else f"Piano Statale: Capacità Nucleare & Indipendenza Uranio ({PAESE})"
axs2[1, 1].set_title(title_nuc, fontsize=11, weight='bold')
axs2[1, 1].set_ylabel("Dipendenza Uranio Estero (%)", color=color_ura)
axs2[1, 1].plot(anni_futuri, scenario_uranio, color=color_ura, lw=3, label="Dipendenza Uranio (Nuclear Dominance)")
axs2[1, 1].fill_between(anni_futuri, ura_low, ura_high, color=color_ura, alpha=0.2)
axs2[1, 1].tick_params(axis='y', labelcolor=color_ura)
axs2[1, 1].set_ylim(0, 100)
axs2[1, 1].grid(alpha=0.3)

ax2_twin2 = axs2[1, 1].twinx()
color_cap = 'tab:purple'
ax2_twin2.set_ylabel("Capacità Nucleare Totale (GW)", color=color_cap)
ax2_twin2.plot(anni_futuri, scenario_capacita_nuc, color=color_cap, lw=3, ls='--', label="Capacità Nucleare (Target 400 GW)")
ax2_twin2.fill_between(anni_futuri, nuc_cap_low, nuc_cap_high, color=color_cap, alpha=0.2)
ax2_twin2.axhline(400.0, color='gray', linestyle=':', label='Target 400 GW 2050')
ax2_twin2.tick_params(axis='y', labelcolor=color_cap)

lines_3, labels_3 = axs2[1, 1].get_legend_handles_labels()
lines_4, labels_4 = ax2_twin2.get_legend_handles_labels()
axs2[1, 1].legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right', fontsize=8)

fig2.tight_layout()

# DASHBOARD 3: Resilienza Nazionale e Net-Zero Tracker
fig3 = plt.figure(figsize=(14, 10))
fig3.suptitle(f"Resilienza Nazionale, Sforzo CAPEX e Net-Zero Tracker ({PAESE})", fontsize=16, fontweight='bold')

# Subplot 1: Net-Zero Tracker (Top Left)
ax_nz = fig3.add_subplot(221)
# Calcolo emissioni baseline di riferimento
base_emissioni = base_consumo_totale * (base_quota_fossile_logistica / 100.0) * PROFILO["emission_factor"]

emissioni_2026 = scenario_emissioni[0]
target_zero = np.linspace(emissioni_2026, 0, len(anni_futuri))

ax_nz.plot(anni_futuri, base_emissioni, color='gray', ls='--', lw=2, label="Baseline Emissioni")
ax_nz.plot(anni_futuri, scenario_emissioni, color='darkred', lw=3, label="Emissioni CO2 (Mediana)")
ax_nz.fill_between(anni_futuri, em_low, em_high, color='red', alpha=0.2, label="Incertezza (95%)")
ax_nz.plot(anni_futuri, target_zero, color='forestgreen', ls='--', lw=2, label="Target Net-Zero (Ideale)")
ax_nz.axvline(x=2026, color='red', linestyle=':', alpha=0.7, label='Inizio Conflitto M.O.')
ax_nz.set_title("Net-Zero Tracker")
ax_nz.set_ylabel("Emissioni (Mt CO2)")
ax_nz.grid(alpha=0.3)
ax_nz.legend(fontsize=8)

# Subplot 2: Radar Chart (Top Right) — 10 metriche identiche allo scorer comparativo
# Le stesse metriche, gli stessi range assoluti: i due radar (singolo e comparativo) sono ora coerenti.
ax_radar = fig3.add_subplot(222, polar=True)

# --- Definizione range assoluti (coerenti con EnergyRaceScorer.METRICHE) ---
_RADAR_METRICHE = [
    # (label_corta, valore_2026, valore_2050, range_min, range_max, higher_better)
]

def _indip_composita_loc(dip_net, dip_ura, prod_pet, exp_gnl, qf, qn, qr):
    # Formula allineata con EnergyRaceScorer._val("indip_geopolitica") in codice_6
    # Pesi: C1=35% C2=25% C3=25% C4=15% — identici al comparatore
    c1 = float(np.clip((-np.clip(dip_net, -50, 100) + 100) / 150 * 100, 0, 100))
    c2 = float(np.clip(100 - dip_ura, 0, 100))
    _p = float(prod_pet) if prod_pet is not None else 0.0
    _g = float(exp_gnl)  if exp_gnl  is not None else 0.0
    c3 = float(np.clip((_p / 15.0) * 50 + (_g / 50.0) * 50, 0, 100))
    _t = max(qf + qn + qr, 1e-6)
    _f, _n, _r = qf / _t, qn / _t, qr / _t
    c4 = float(np.clip((1 - (_f**2 + _n**2 + _r**2)) / (1 - 1/3) * 100, 0, 100))
    # Penalità ME aggiuntiva (non presente in codice_6 — abbassa leggermente il punteggio
    # quando uranio >70% e conflitto attivo, per coerenza narrativa con lo scenario)
    _ura_penalty = float(np.clip((dip_ura - 70) / 30 * intensita_conflitto_me * 10, 0, 10))
    score = 0.35 * c1 + 0.25 * c2 + 0.25 * c3 + 0.15 * c4 - _ura_penalty
    return float(np.clip(score, 0, 100))

_idx_2033 = np.where(anni_futuri == 2033)[0]
_ura_2026 = float(scenario_uranio[0])
_ura_2050 = float(scenario_uranio[-1])
_idx_2033 = np.where(anni_futuri == 2033)[0]
_smr_idx_2040 = np.where(anni_futuri == 2040)[0]
_smr_2026 = 0.0
_smr_2050 = float(scenario_smr_capacity[_smr_idx_2040[0]]) if len(_smr_idx_2040) else float(scenario_smr_capacity[-1])
_diffs_fos = np.diff(scenario_quota_fossile)
_fossil_speed = float(-np.mean(_diffs_fos[_diffs_fos < 0])) if np.any(_diffs_fos < 0) else 0.0
_pet_2026 = float(scenario_prod_petrolio[0])
_pet_2050 = float(scenario_prod_petrolio[-1])
_gnl_2026 = float(scenario_export_gnl[0])
_gnl_2050 = float(scenario_export_gnl[-1])
_capex_max = max(float(np.max(scenario_capex_pil)) * 1.2, 0.5)
_em_tot = float(np.median(mc_emissioni_cum)) / 1000

_indip_2026 = _indip_composita_loc(scenario_dipendenza[0], _ura_2026, _pet_2026, _gnl_2026,
                                    scenario_quota_fossile[0]/100, scenario_quota_nucleare[0]/100, scenario_quota_rinnovabile[0]/100)
# Per _indip_2050 usiamo il valore uranio al 2033 (più conservativo e verificabile)
# invece del 2050 proiettato che era troppo ottimistico (DPA portava a 20%)
_ura_per_indip_2050 = float(scenario_uranio[_idx_2033[0]]) if len(_idx_2033) else _ura_2050
_indip_2050 = _indip_composita_loc(scenario_dipendenza[-1], _ura_per_indip_2050, _pet_2050, _gnl_2050,
                                    scenario_quota_fossile[-1]/100, scenario_quota_nucleare[-1]/100, scenario_quota_rinnovabile[-1]/100)


# Radar: 6 metriche — aggiunta indipendenza uranio esplicita dai risultati MC
# Usa scenario_uranio già calcolato dal loop Monte Carlo

# 1. Non-fossile nel mix (% diretta)
_nf_2026 = float(100 - scenario_quota_fossile[0])
_nf_2050 = float(100 - scenario_quota_fossile[-1])

# 2. Quota nucleare % nel mix
_qnuc_2026 = float(np.clip(scenario_quota_nucleare[0], 0, 100))
_qnuc_2050 = float(np.clip(scenario_quota_nucleare[-1], 0, 100))

# 3. Indipendenza geopolitica composita (aggiornata con peso uranio 35%)
_indip_r_2026 = float(np.clip(_indip_2026, 0, 100))
_indip_r_2050 = float(np.clip(_indip_2050, 0, 100))

# 4. Indipendenza uranio — inversa della dipendenza estera dal grafico MC
# Usa il valore 2033 come punto di arrivo (obiettivo "Nuclear Dominance 3 by 33")
_ura_indip_2026 = float(np.clip(100 - scenario_uranio[0], 0, 100))
_ura_indip_2050 = float(np.clip(100 - scenario_uranio[_idx_2033[0]], 0, 100)) if len(_idx_2033) else float(np.clip(100 - scenario_uranio[-1], 0, 100))

# 5. Elettrificazione (% diretta)
_el_2026 = float(np.clip(scenario_elettrificazione[0], 0, 100))
_el_2050 = float(np.clip(scenario_elettrificazione[-1], 0, 100))

# 6. Efficienza energetica — inversa intensità normalizzata
_int_max = float(np.max(intensita_storica)) if len(intensita_storica) > 0 else 3.5
_eff_2026 = float(np.clip((1 - scenario_intensita[0] / _int_max) * 100, 0, 100))
_eff_2050 = float(np.clip((1 - scenario_intensita[-1] / _int_max) * 100, 0, 100))

categorie_radar = ['Non-Fossile\n%', 'Quota\nNucleare', 'Indip.\nGeopol.', 'Indip.\nUranio', 'Elettrific.', 'Efficienza\nEnerg.']
N = len(categorie_radar)
valori_2023 = [_nf_2026,  _qnuc_2026, _indip_r_2026, _ura_indip_2026, _el_2026, _eff_2026]
valori_2050 = [_nf_2050,  _qnuc_2050, _indip_r_2050, _ura_indip_2050, _el_2050, _eff_2050]

print("\n=== RADAR VALORI (scala 0-100 diretta) ===")
for cat, v26, v50 in zip(categorie_radar, valori_2023, valori_2050):
    print(f"  {cat.replace(chr(10),' '):<22} 2026={v26:5.1f}  2050={v50:5.1f}  diff={v50-v26:+.1f}")
print("============================================\n")

N = len(categorie_radar)

angoli = np.linspace(0, 2*np.pi, N, endpoint=False)
angoli_c = np.concatenate([angoli, [angoli[0]]])
v23_c = valori_2023 + [valori_2023[0]]
v50_c = valori_2050 + [valori_2050[0]]

for r in [25, 50, 75, 100]:
    ax_radar.plot(angoli_c, [r]*(N+1), color='#444', lw=0.5)
for ang in angoli:
    ax_radar.plot([ang,ang],[0,100], color='#555', lw=0.4, alpha=0.4)

ax_radar.plot(angoli_c, v23_c, color='gray', lw=2, ls='dashed', label='Fotografia 2026')
ax_radar.fill(angoli_c, v23_c, color='gray', alpha=0.1)
ax_radar.plot(angoli_c, v50_c, color='dodgerblue', lw=2.5, ls='solid', label='Mediana 2050')
ax_radar.fill(angoli_c, v50_c, color='dodgerblue', alpha=0.3)

ax_radar.set_xticks([])
for i, (ang, cat) in enumerate(zip(angoli, categorie_radar)):
    _r = 122 + (4 if i % 2 == 0 else -2)
    _ha = 'left' if np.sin(ang) > 0.15 else ('right' if np.sin(ang) < -0.15 else 'center')
    ax_radar.text(ang, _r, cat, ha=_ha, va='center', color='white', fontsize=7, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.15', fc='#1a1a2e', ec='none', alpha=0.6))
ax_radar.set_ylim(0, 145)
ax_radar.set_yticks([25,50,75])
ax_radar.set_yticklabels(['25','50','75'], color='grey', size=7)
ax_radar.legend(loc='upper right', bbox_to_anchor=(0.15, 0.12), fontsize=8)

# Aggiorna valori esportati nel npz (coerenti con le 10 metriche)
valori_2023 = valori_2023   # già lista di 10 valori
valori_2050 = valori_2050   # già lista di 10 valori

# Subplot 3: Sforzo CAPEX su PIL (%) con evidenziazione della Bolla Speculativa (Bottom Left)
ax_capex = fig3.add_subplot(223)
ax_capex.plot(anni_futuri, scenario_capex_pil, color='indigo', lw=3, label="CAPEX/PIL Mediano")
ax_capex.fill_between(anni_futuri, capex_low, capex_high, color='indigo', alpha=0.2, label="Incertezza (95%)")
# Evidenziazione della fase calda di bolla inflazionistica centrata sul 2027
ax_capex.fill_between(anni_futuri, scenario_capex_pil, where=((anni_futuri >= 2026) & (anni_futuri <= 2029)),
                      color='orange', alpha=0.2, label='Fase Bolla Speculativa (Shiller)')
# Annota il picco nel 2027
picco_anno = 2027
picco_val = scenario_capex_pil[anni_futuri == picco_anno][0]
ax_capex.annotate(f'Picco Bolla Shiller: {picco_val:.2f}%',
                  xy=(picco_anno, picco_val),
                  xytext=(picco_anno + 1.5, picco_val + 0.6),
                  arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6, headlength=6),
                  fontsize=9, weight='bold')
ax_capex.set_title("Sforzo Economico di Transizione (CAPEX / PIL %)", fontsize=11, weight='bold')
ax_capex.set_ylabel("% PIL")
ax_capex.set_xlabel("Anno")
ax_capex.grid(alpha=0.3)
ax_capex.legend(fontsize=8, loc='upper right')

# Subplot 4: Indice Speculativo di Shiller - Moltiplicatore Costi (Bottom Right)
ax_shiller = fig3.add_subplot(224)
ax_shiller.plot(anni_futuri, moltiplicatore_capex, color='crimson', lw=3, label='Moltiplicatore CAPEX (Shiller Bubble)')
ax_shiller.fill_between(anni_futuri, 1.0, moltiplicatore_capex, color='crimson', alpha=0.15)
ax_shiller.axhline(1.0, color='gray', linestyle='--', label='Costo Base Lineare')
ax_shiller.set_title("Indice Speculativo di Shiller (Asset Inflation)", fontsize=11, weight='bold')
ax_shiller.set_ylabel("Moltiplicatore Costo")
ax_shiller.set_xlabel("Anno")
ax_shiller.grid(alpha=0.3)
ax_shiller.legend(fontsize=8, loc='upper right')

fig3.tight_layout()

# ==========================================
# DASHBOARD 4: Black Swan Analysis
# ==========================================
# Mostra come gli eventi estremi allargano il CI rispetto a un mondo "ordinato"
# Confronto mediana con/senza BSE non e' implementato (richiederebbe doppio MC),
# ma il fan chart del 5-95% mostra gia' l'asimmetria prodotta dagli eventi estremi.
fig4, axs4 = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle(f"Black Swan Analysis — Impatto Eventi Estremi ({PAESE})", fontsize=16, fontweight='bold')

# Subplot 1: Fan chart fossile con percentili 5-25-50-75-95
ax_fan = axs4[0, 0]
p05_f, p25_f, p50_f, p75_f, p95_f = np.percentile(mc_quota_fossile, [5, 25, 50, 75, 95], axis=0)
ax_fan.fill_between(anni_futuri, p05_f, p95_f, color='purple', alpha=0.15, label='CI 90%')
ax_fan.fill_between(anni_futuri, p25_f, p75_f, color='purple', alpha=0.30, label='CI 50%')
ax_fan.plot(anni_futuri, p50_f, color='purple', lw=2.5, label='Mediana')
ax_fan.plot(anni_storici, quota_fossile, 'k-', lw=2, label='Storico')
ax_fan.axvline(2026, color='red', ls=':', alpha=0.7)
ax_fan.set_title("Fan Chart Quota Fossili (con Black Swan)")
ax_fan.set_ylabel("%")
ax_fan.set_ylim(0, 100)
ax_fan.grid(alpha=0.3)
ax_fan.legend(fontsize=8)

# Subplot 2: Fan chart capacita' nucleare
ax_nuc_fan = axs4[0, 1]
p05_n, p25_n, p50_n, p75_n, p95_n = np.percentile(mc_capacita_nucleare, [5, 25, 50, 75, 95], axis=0)
ax_nuc_fan.fill_between(anni_futuri, p05_n, p95_n, color='orange', alpha=0.15, label='CI 90%')
ax_nuc_fan.fill_between(anni_futuri, p25_n, p75_n, color='orange', alpha=0.30, label='CI 50%')
ax_nuc_fan.plot(anni_futuri, p50_n, color='orange', lw=2.5, label='Mediana')
ax_nuc_fan.axhline(PROFILO["obiettivo_nucleare_2050"], color='gray', ls=':', label=f'Target {PROFILO["obiettivo_nucleare_2050"]:.0f} GW')
ax_nuc_fan.axvline(2026, color='red', ls=':', alpha=0.7)
ax_nuc_fan.set_title("Fan Chart Capacità Nucleare (con Black Swan)")
ax_nuc_fan.set_ylabel("GW")
ax_nuc_fan.grid(alpha=0.3)
ax_nuc_fan.legend(fontsize=8)

# Subplot 3: Frequenza eventi Black Swan
ax_bs_freq = axs4[1, 0]
cats_labels = [c.replace('_', '\n') for c in registro_bs.keys()]
freqs = [registro_bs[c] / N_SIMULAZIONI * 100 for c in registro_bs.keys()]
colors_bs = ['#d62728', '#e377c2', '#8c564b', '#17becf', '#bcbd22']
bars = ax_bs_freq.bar(range(len(cats_labels)), freqs, color=colors_bs, alpha=0.8)
ax_bs_freq.set_xticks(range(len(cats_labels)))
ax_bs_freq.set_xticklabels(cats_labels, fontsize=8)
ax_bs_freq.set_ylabel("% dei run MC con ≥1 evento")
ax_bs_freq.set_title("Frequenza Attivazione Black Swan (1000 run)")
ax_bs_freq.set_ylim(0, 100)
for bar, freq in zip(bars, freqs):
    ax_bs_freq.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{freq:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax_bs_freq.grid(alpha=0.3, axis='y')

# Subplot 4: Distribuzione emissioni cumulative (istogramma) — mostra le code grasse
ax_em_hist = axs4[1, 1]
ax_em_hist.hist(mc_emissioni_cum / 1000, bins=50, color='darkred', alpha=0.7, edgecolor='white', lw=0.5)
ax_em_hist.axvline(np.median(mc_emissioni_cum) / 1000, color='white', lw=2, ls='--', label='Mediana')
ax_em_hist.axvline(np.percentile(mc_emissioni_cum, 5) / 1000, color='lime', lw=1.5, ls=':', label='P5 (scenario ottimistico)')
ax_em_hist.axvline(np.percentile(mc_emissioni_cum, 95) / 1000, color='orange', lw=1.5, ls=':', label='P95 (scenario pessimistico)')
ax_em_hist.set_title("Distribuzione Emissioni Cumulative 2026-2050 (Gt CO2)")
ax_em_hist.set_xlabel("Gt CO2")
ax_em_hist.set_ylabel("Frequenza (run MC)")
ax_em_hist.legend(fontsize=8)
ax_em_hist.grid(alpha=0.3)

# Annota skewness: coda destra = black swan pessimistici piu' impattanti di quelli ottimistici?
_skew = float(np.mean(((mc_emissioni_cum - np.mean(mc_emissioni_cum)) / np.std(mc_emissioni_cum))**3))
ax_em_hist.text(0.97, 0.95, f'Skewness: {_skew:+.2f}',
                transform=ax_em_hist.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

fig4.tight_layout()

# ==========================================
# 8. EXPORT NPZ — snapshot per il modulo comparativo
# ==========================================
# Salva tutti i dati necessari per EnergyRaceComparator in un file .npz
# Convenzione nome: risultati_UnitedStates.npz / risultati_China.npz
_paese_safe = PAESE.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
_npz_path = f"risultati_{_paese_safe}.npz"

np.savez_compressed(
    _npz_path,
    # --- Identificatori ---
    paese           = np.array([PAESE]),
    intensita_conflitto = np.array([intensita_conflitto_me]),

    # --- Assi temporali ---
    anni_storici    = anni_storici,
    anni_futuri     = anni_futuri,

    # --- Storici (per grafici di continuità) ---
    quota_fossile_storico    = quota_fossile,
    quota_rinnovabile_storico= quota_rinnovabile,
    quota_nucleare_storico   = quota_nucleare_storica,
    consumo_totale_storico   = consumo_totale,
    intensita_storica        = intensita_storica,
    dipendenza_storica       = dipendenza_storica,
    dip_verde_storica        = dip_verde_storica,
    energia_per_capita_storica = energia_per_capita_storica,

    # --- Scenari mediani ---
    scenario_quota_fossile      = scenario_quota_fossile,
    scenario_quota_nucleare     = scenario_quota_nucleare,
    scenario_quota_rinnovabile  = scenario_quota_rinnovabile,
    scenario_consumo            = scenario_consumo,
    scenario_intensita          = scenario_intensita,
    scenario_dipendenza         = scenario_dipendenza,
    scenario_dip_verde          = scenario_dip_verde,
    scenario_emissioni          = scenario_emissioni,
    scenario_elettrificazione   = scenario_elettrificazione,
    scenario_capex_pil          = scenario_capex_pil,
    scenario_capacita_nuc       = scenario_capacita_nuc,
    scenario_smr_capacity       = scenario_smr_capacity,
    scenario_uranio             = scenario_uranio,
    scenario_prod_petrolio      = scenario_prod_petrolio,
    scenario_export_gnl         = scenario_export_gnl,
    energia_per_capita_futura   = energia_per_capita_futura,
    consumo_fossile_twh         = consumo_fossile_twh,
    consumo_nucleare_twh        = consumo_nucleare_twh,
    consumo_rinnovabile_twh     = consumo_rinnovabile_twh,

    # --- Bande di incertezza (CI 95%) ---
    sqf_low  = sqf_low,   sqf_high  = sqf_high,
    sqn_low  = sqn_low,   sqn_high  = sqn_high,
    sqr_low  = sqr_low,   sqr_high  = sqr_high,
    sc_low   = sc_low,    sc_high   = sc_high,
    em_low   = em_low,    em_high   = em_high,
    nuc_cap_low = nuc_cap_low, nuc_cap_high = nuc_cap_high,
    ura_low  = ura_low,   ura_high  = ura_high,
    pet_low  = pet_low,   pet_high  = pet_high,

    # --- Distribuzioni MC complete (per statistiche avanzate nel comparatore) ---
    mc_emissioni_cum        = mc_emissioni_cum,
    mc_quota_fossile_p05    = np.percentile(mc_quota_fossile,    5, axis=0),
    mc_quota_fossile_p95    = np.percentile(mc_quota_fossile,   95, axis=0),
    mc_capacita_nuc_p05     = np.percentile(mc_capacita_nucleare, 5, axis=0),
    mc_capacita_nuc_p95     = np.percentile(mc_capacita_nucleare,95, axis=0),
    mc_emissioni_p05        = np.percentile(mc_emissioni,         5, axis=0),
    mc_emissioni_p95        = np.percentile(mc_emissioni,        95, axis=0),
    mc_rinnovabile_p05      = np.percentile(mc_quota_rinnovabile, 5, axis=0),
    mc_rinnovabile_p95      = np.percentile(mc_quota_rinnovabile,95, axis=0),

    # --- Radar scorecard (valori già calcolati, 6 metriche) ---
    radar_valori_2026       = np.array(valori_2023),
    radar_valori_2050       = np.array(valori_2050),
    radar_categorie         = np.array(categorie_radar),
    mediana_emissioni_cum   = np.array([mediana_emissioni_cum]),

    # --- Black Swan stats ---
    bs_registro_keys        = np.array(list(registro_bs.keys())),
    bs_registro_valori      = np.array(list(registro_bs.values())),
    n_simulazioni           = np.array([N_SIMULAZIONI]),
)
print(f"\n[EXPORT] Snapshot salvato in: {_npz_path}")
print(f"  Variabili esportate: {len(np.load(_npz_path, allow_pickle=True).files)}")

print("\nDashboard pronti per essere salvati!")
fig4.savefig('fig4_blackswan.png')
fig3.savefig('fig3.png')
fig2.savefig('fig2.png')
fig1.savefig('fig1.png')
print("Salvataggio completato: fig1.png, fig2.png, fig3.png, fig4_blackswan.png")
plt.show()
# TO_REMOVE
# This file is a one-off script / dev utility and is no longer needed.
# Safe to delete after confirming no active references.
# -------------------------------------------------------------------
"""
Update PostgreSQL: ganti data AADI/ADRO/PTBA dengan BBKP + TINS
"""
import psycopg2

DB_URL = (
    "postgresql://neondb_owner:npg_oL9hyFqOPEB3"
    "@ep-broad-glitter-a45az27j-pooler.us-east-1.aws.neon.tech"
    "/neondb?sslmode=require"
)

conn = psycopg2.connect(DB_URL)
conn.autocommit = False
cur = conn.cursor()

try:
    # 1. Hapus data lama (FK constraint: notes dulu, baru watchlist)
    print("Menghapus data lama...")
    cur.execute("DELETE FROM analyst_notes")
    print(f"  analyst_notes: {cur.rowcount} baris dihapus")
    cur.execute("DELETE FROM company_watchlist")
    print(f"  company_watchlist: {cur.rowcount} baris dihapus")

    # 2. Insert company_watchlist: BBKP + TINS
    print("\nInsert company_watchlist...")
    cur.execute("""
        INSERT INTO company_watchlist (
            ticker, company_name, sector, exchange,
            report_period,
            revenue_usd_k, net_profit_usd_k,
            gross_margin_pct, debt_status,
            last_reviewed_by, review_date, watch_status
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        'BBKP', 'PT Bank KB Bukopin Tbk (KB Bank)', 'Banking', 'IDX',
        'Q1-2025',
        184.000,   # Pendapatan bunga bersih bank-only (IDR miliar)
        352.000,   # Laba bersih konsolidasi (IDR miliar)
        1.090,     # NIM (%)
        'improving: NPL 9.10% turun, LAR 23.41% turun',
        1, '2026-04-15', 'active'
    ))
    cur.execute("""
        INSERT INTO company_watchlist (
            ticker, company_name, sector, exchange,
            report_period,
            revenue_usd_k, net_profit_usd_k,
            gross_margin_pct, debt_status,
            last_reviewed_by, review_date, watch_status
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        'TINS', 'PT Timah Tbk', 'Basic Materials - Tin Mining', 'IDX',
        'Q1-2025',
        2100.000,  # Revenue Q1 2025 (IDR miliar, +2.1% YoY)
        116.860,   # Laba bersih Q1 2025 (IDR miliar, 120% dari target)
        18.095,    # Gross margin % = (2100-1720)/2100*100
        'deleveraging: liabilitas -9%, DER 63.5%, CR 238.7%',
        2, '2026-04-15', 'active'
    ))
    print("  BBKP + TINS inserted")

    # 3. Insert analyst_notes: BBKP + TINS
    print("\nInsert analyst_notes...")
    cur.execute("""
        INSERT INTO analyst_notes (
            analyst_id, ticker, report_period,
            revenue_trend, revenue_change_pct, gross_margin_pct,
            key_highlights, risk_notes, recommendation, risk_level,
            presentation_strategy
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        1, 'BBKP', 'Q1-2025',
        'growing', 11.19, 1.090,
        ('Laba bersih konsolidasi Q1 2025 Rp352 miliar (vs rugi Rp827 miliar Q1 2024). '
         'NIM naik dari 0.94% ke 1.09%. DPK Rp43.83 triliun (+10.86% YoY). '
         'CASA Rp12.38 triliun (+16.83% YoY). NPL gross 9.10% (dari 9.92%). '
         'LAR 23.41% (dari 34.33%). Kredit tumbuh: retail +22.68%, korporasi +12.14%, UMKM +3.29%. '
         'Migrasi core banking ke NGBS selesai Q1 2025. KB Kookmin Bank Korea pemegang saham 66.88%.'),
        ('NPL gross 9.10% masih di atas rata-rata industri perbankan. '
         'LAR 23.41% perlu monitoring lanjutan. '
         'Ketergantungan funding dari induk KB Kookmin Korea.'),
        'buy', 'medium',
        ('Highlight transformasi: dari rugi Rp827M ke laba Rp352M. '
         'Tekankan selesainya migrasi NGBS sebagai fondasi akselerasi digital. '
         'NIM improvement sebagai indikator efisiensi funding ke klien institusional.')
    ))
    cur.execute("""
        INSERT INTO analyst_notes (
            analyst_id, ticker, report_period,
            revenue_trend, revenue_change_pct, gross_margin_pct,
            key_highlights, risk_notes, recommendation, risk_level,
            presentation_strategy
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        2, 'TINS', 'Q1-2025',
        'growing', 2.10, 18.095,
        ('Laba bersih Q1 2025 Rp116.86 miliar (120% dari target Rp97.46 miliar). '
         'Revenue Rp2.10 triliun (+2.1% YoY). Beban pokok Rp1.72 triliun (-2.6% YoY). '
         'Laba usaha Rp148 miliar (vs Rp93M Q1 2024). EBITDA Rp384 miliar (+14% YoY). '
         'Total aset Rp12.49 triliun. Liabilitas Rp4.85 triliun (-9%). Ekuitas Rp7.64 triliun (+3%). '
         'DER 63.5%. Current Ratio 238.7%. Quick Ratio 66.1%. '
         'Produksi bijih timah 3.215 ton Sn (-40% YoY). Produksi logam 3.095 MT (-31%). '
         'Harga jual rata-rata USD32.495/MT (+20%). Harga LME Q1 2025 USD31.804/MT (+21.2% YoY). '
         'Ekspor: Korea 19%, Jepang 19%, Singapura 14%, Belanda 11%.'),
        ('Penurunan produksi 40% perlu dipantau. '
         'Sustainability strategi high-price low-volume belum terkonfirmasi di Q2-Q3. '
         'Konsentrasi ekspor Korea+Jepang sekitar 38% -- risiko konsentrasi pasar.'),
        'hold', 'medium',
        ('Framing strategi high-price low-volume Q1 2025. '
         'Produksi turun 40% tapi profitabilitas meningkat karena harga LME naik 21.2% YoY. '
         'Monitor recovery produksi di Q2-Q3 sebelum upgrade rekomendasi ke BUY.')
    ))
    print("  BBKP + TINS inserted")

    conn.commit()
    print("\n✓ Commit berhasil!")

except Exception as e:
    conn.rollback()
    print(f"\n✗ Error — rollback: {e}")
    raise
finally:
    cur.close()
    conn.close()

# Verifikasi
print("\n=== VERIFIKASI ===")
conn2 = psycopg2.connect(DB_URL)
cur2  = conn2.cursor()

cur2.execute("SELECT ticker, company_name, net_profit_usd_k, gross_margin_pct, debt_status FROM company_watchlist")
print("company_watchlist:")
for r in cur2.fetchall():
    print(f"  {r[0]:<6} | {r[1]:<40} | laba={r[2]} | NIM/margin={r[3]}%")
    print(f"         debt_status: {r[4]}")

cur2.execute("""
    SELECT an.ticker, up.name, up.position, an.recommendation, an.risk_level,
           LEFT(an.key_highlights, 80) AS highlights
    FROM analyst_notes an
    JOIN user_profiles up ON an.analyst_id = up.id
""")
print("\nanalyst_notes JOIN user_profiles:")
for r in cur2.fetchall():
    print(f"  {r[0]:<6} | {r[1]:<20} ({r[2]}) | {r[3]} / {r[4]}")
    print(f"         {r[5]}...")

cur2.close()
conn2.close()
print("\nSelesai. Database siap untuk Skenario B.")

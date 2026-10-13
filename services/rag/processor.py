"""The RAG engine: PDFQAProcessor and its shared instance `processor`.

The methods live next to this module, one mixin per concern: routing,
indexes, document/chat/public-link/structured-data search, hybrid ranking,
answers, LLMs and the gap check. This module composes them."""
import logging
import threading
from typing import Dict, Tuple

from config import config
from services.rag.index_cache import IndexCache
from services.rag.routing import RoutingMixin
from services.rag.indexes import IndexMixin
from services.rag.document_search import DocumentSearchMixin
from services.rag.chat_search import ChatSearchMixin
from services.rag.public_link_search import PublicLinkSearchMixin
from services.rag.llm import LLMMixin
from services.rag.structured_data import StructuredDataMixin
from services.rag.hybrid import HybridSearchMixin
from services.rag.answer import AnswerMixin
from services.rag.gap_check import GapCheckMixin

# Re-exported: tests import these from here.
from services.rag.shared import OFF_TOPIC_REDIRECT_ID, OFF_TOPIC_INSTRUCTION_ID, OFF_TOPIC_REDIRECT_EN, OFF_TOPIC_INSTRUCTION_EN  # noqa: F401

logger = logging.getLogger(__name__)


class PDFQAProcessor(
    RoutingMixin,
    IndexMixin,
    DocumentSearchMixin,
    ChatSearchMixin,
    PublicLinkSearchMixin,
    LLMMixin,
    StructuredDataMixin,
    HybridSearchMixin,
    AnswerMixin,
    GapCheckMixin,
):
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.embeddings = None
        # Loaded FAISS indexes, PDF and chat ("chat_<id>"), within a memory
        # budget; a BM25 index is dropped together with its collection's index.
        self.bm25_cache = {}
        self.vector_store_cache = IndexCache(
            config.index_cache_max_bytes,
            on_evict=lambda key: self.bm25_cache.pop(key, None),
        )
        # collection_id -> (time.monotonic() before which it's skipped,
        # consecutive failed loads); see get_vector_store.
        self._missing_vector_stores: Dict[str, Tuple[float, int]] = {}
        self._cache_lock = threading.RLock()
        self._initialized = False
        self._init_lock = threading.Lock()
        self.db_manager = None
        self._db_initialized = False
        
        # Multi-LLM support
        self._llm_cache = {}  # Cache for loaded LLMs: {provider_model: llm_instance}
        self._current_provider = None
        self._current_model = None

        self.query_expansion_terms = {
            "apa itu": ["definisi", "pengertian", "arti", "makna", "jelaskan"],
            "proses": ["tahapan", "langkah", "mekanisme", "cara kerja"],
            "auction": ["lelang", "penawaran", "bidding", "tender"],
            # LPDU document keywords
            "buyback": ["pembelian kembali", "buyback cash", "buyback debt switch"],
            "lpdu": ["layanan perdagangan dealer utama", "dealer utama"],
            "lpksbn": ["lelang pembelian kembali surat berharga negara"],
            "settlement": ["penyelesaian transaksi", "setelmen"],
            "staple bonds": ["paket staple", "destination series", "source series"],
            "quotation": ["kuotasi", "penawaran", "quote"],
            "allocation": ["alokasi", "pemenang lelang"]
        }

        # Add database-related query expansion
        self.db_query_expansion_terms = {
            "user": ["pengguna", "karyawan", "staff", "employee", "profil"],
            "product": ["produk", "barang", "item", "inventory"],
            "order": ["pesanan", "pembelian", "transaksi", "orderan", "pemesanan"],
            "price": ["harga", "cost", "biaya", "tarif", "nilai"],
            "quantity": ["jumlah", "kuantitas", "banyak", "stock"]
        }
        
        # Document-specific keywords (for PDF search priority)
        self.pdf_keywords = [
            'lpdu', 'lpksbn', 'buyback', 'debt switch', 'auction', 'lelang',
            'sun', 'sbn', 'djppr', 'mofids', 'dealer utama', 'settlement',
            'quotation', 'kuotasi', 'alokasi', 'staple bonds', 'securities',
            'fungsional', 'persyaratan', 'kode a', 'lpdu-bcds', 'lpdu-sa', 'lpdu-dssb',
            'maker checker', 'enrich data', 'plte', 'bank indonesia',
            'sop', 'prosedur', 'standar', 'approval', 'deployment', 'testing',
            'development', 'sdlc', 'requirement', 'design', 'maintenance'
        ]

        # Smart table routing - map keywords to specific tables
        # NOTE: Use generic keywords, NOT hardcoded values like names
        self.table_keywords = {
            "user_profiles": [
                'user', 'pengguna', 'karyawan', 'staff', 'employee', 'profil',
                'nama', 'email', 'department', 'position', 'jabatan', 'pegawai',
                'siapa', 'orang', 'anggota', 'member', 'kontak', 'telepon', 'phone',
                'divisi', 'departemen', 'bagian'  # department-related
            ],
            "products": [
                'product', 'produk', 'barang', 'item', 'harga', 'price',
                'stock', 'stok', 'kategori', 'category', 'jual', 'beli',
                'license', 'software', 'tools', 'service'
            ],
            "orders": [
                'order', 'pesanan', 'pembelian', 'transaksi', 'beli', 'pesan',
                'status', 'pending', 'completed', 'shipped', 'quantity',
                'total', 'amount', 'tanggal', 'invoice', 'processing', 'proyek'
            ]
        }
        
        # Person name patterns (untuk deteksi nama orang tanpa hardcode)
        self.person_question_patterns = [
            'siapa', 'who is', 'nama', 'karyawan bernama', 'user bernama',
            'cari orang', 'find person', 'profile', 'kontak', 'contact',
            'email', 'telepon', 'phone'
        ]
        
        # Tambahkan patterns untuk intent detection yang lebih baik
        self.intent_patterns = {
            'data_retrieval': [
                r'berapa\s+(harga|jumlah|total|stock)',
                r'(harga|price)\s+[a-zA-Z0-9\s]+',
                r'siapa\s+yang',
                r'(cari|tampilkan|lihat)\s+(data|informasi)',
                r'^[a-zA-Z]+\s+\d+$',  # Pattern seperti "Laptop 5"
            ],
            'comparison': [
                r'(bandingkan|perbandingan|vs\.?)',
                r'(lebih|paling)\s+(murah|mahal|banyak|sedikit)',
                r'mana\s+yang\s+(lebih|paling)',
            ],
            'aggregation': [
                r'(total|jumlah|rata-rata|rerata|average|sum)\s+[a-zA-Z]',
                r'berapa\s+total',
                r'hitung\s+(total|jumlah)',
                r'(semua|semua\s+data)\s+[a-zA-Z]',
            ],
            'explanation': [
                r'apa\s+itu',
                r'jelaskan\s+tentang',
                r'definisi\s+dari',
                r'bagaimana\s+cara',
                r'proses\s+[a-zA-Z]',
            ]
        }

        # Known "/" commands (must mirror chat-ui's SLASH_COMMANDS). Backend
        # guard only — the frontend intercepts unmatched slash input before
        # it ever reaches this API, this exists for direct API calls/bypass.
        self.known_slash_commands = {"/gap-check", "/collections", "/history", "/upload", "/help"}


processor = PDFQAProcessor()

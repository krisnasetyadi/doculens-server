Available online at https://teewanjournal.com/index.php/juragan
eISSN: 3063 - 802X ; pISSN: 3063 - 8011
ARTICLE INFO (^) ABSTRACT
Article History: Key personnel turnover creates knowledge gaps in document-based service
organizations, where information is distributed across technical
specifications, operational databases, and team discussions. This study
develops a multi-source Retrieval-Augmented Generation (RAG)based
Question Answering (QA) system that automatically integrates
heterogeneous knowledge sources through a unified source parameter.
Using the Adapter Pattern, the system converts PDF/TXT documents and
PostgreSQL tables into a common representation, builds a FAISS vector
index, retrieves relevant context, and generates grounded answers with
Gemini 2.5 Flash. Evaluation employs eight metrics and three composite
scores: Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval
Score (MSRS), and Answer Quality Index (AQI). Experiments were
conducted on the BOND_SYS dataset using 25 Indonesian questions
covering specification documents, an 8-table PostgreSQL database, and 908
developer discussion messages. Results show perfect retrieval performance
(Precision@K = 1.000; MRR = 1.000) across all scenarios. The full hybrid
configuration achieves the highest Overall score (0.373), while Scenario C
records the highest MSRS (0.825). Scenario E obtains ROUGE-L = 0.
and BLEU-1 = 0.196 using five manually curated reference answers. Two
baseline comparisons further support this contribution: a zero-shot LLM
without retrieval correctly answered only 8% of questions, while a BM
keyword-search baseline, competitive on single-source scenarios, was
outperformed on cross-referencing tasks, underscoring the added value of
dense multi-source retrieval. The findings demonstrate that integrating
formal documents, structured databases, and discussion logs enhances
knowledge transfer and question answering for organizational support and
employee onboarding.
Submitted 10 - 07 - 2026
Received -
Published -
Keywords:
Retrieval-Augmented
Generation;
Question Answering;
Multi-Source;
FAISS;
Knowledge Transfer;
(^)
(^) Vol. 3, No. 2 , 2026

Hal. 01- 10
(^)

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
INTRODUCTION
Organizations operating in document-based service ecosystems, such as financial
institutions, domain-specific technical service providers, and transaction platform operators, rely
heavily on knowledge accumulated across various artifacts: functional requirement
specifications, operational procedures, system configurations, and transaction records. A critical
challenge arises when key personnel turnover occurs: domain knowledge embedded in
individuals must be transferred to successors within a limited time, while service operations
cannot be interrupted. According to Chui et al. (2012), knowledge workers spend approximately
20% of their working time searching for internal information, and IDC (2023) reports that up to
90% of organizational data is unstructured scattered across PDF documents, text files, and
databases, such that only a small fraction is effectively accessible and utilized.
This problem is further compounded by the heterogeneous distribution of organizational
knowledge: functional specifications are stored in PDF documents spanning tens to hundreds of
pages, operational data resides in relational database tables, while the context of technical
decision often exists only in the personal memory of the individual involved. Nonaka and
Takeuchi (1995) distinguish between tacit knowledge (implicit in an individual’s mind).
However, even explicit knowledge is difficult to locate quickly when it is scattered across
repositories in different formats. Alavi and Leidner (2001) further emphasize that knowledge
management systems often fail when organizational knowledge remains fragmented across
repositories and cannot be operationalized in day-to-day decision workflows. Gartner (2020)
estimates that employees spend up to 30% of their working time on low-value activities that could
be automated, including document retrieval and repetitive operational information clarification.
In the enterprise environment that forms the context of this study, a financial instrument
transaction platform manages the offering and allocation processes of financial instruments for
registered participant institutions. The impact of turnover is felt directly at the operational level.
Over a two-year period, two consecutive changes of Product Owner occurred, with the team size
shrinking from three to two members. As a result, each technical clarification request from
external users (platform participants), such as transaction process flows, configuration
parameters, or instrument rules, required response times ranging from several hours to a full
working day, depending on how thoroughly the context was documented and how familiar the
remaining personnel were with the material. This condition represents a common pattern found
across various document-based service organizations: the volume of knowledge does not
decrease, but the human capacity to access it quickly and accurately becomes increasingly limited.
Traditional knowledge management (KM) solutions such as internal wikis and static
knowledge bases are unable to answer dynamic questions that require real-time cross-source
inference. Users must still manually browse through tens to hundreds of pages of technical
specifications and operational procedures while simultaneously checking configuration tables in
the database; a process that consumes hours or even days before a conclusion can be formulated.

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Large Language Models (LLMs) without factual grounding present a different risk: Gao et al.
(2024) identify three critical weaknesses of LLMs in domain-specific service context namely
hallucination, outdated knowledge and non-transparent reasoning, while Ji et al (2023) provide
broader evidence that hallucination remains a structural limitation in neutral text generation
systems. Furthermore, the Self-RAG approach (Asai et al. 2023) explicitly trains LLMs to evaluate
retrieval relevance and self-critique their own answer, confirming that standard LLMs without
such mechanisms lend to produce answer without considering the boundaries of their
knowledge, a behavior that is particularly dangerous in service contexts demanding accuracy and
traceability. A clear gap therefore exists: no solution is yet capable of answering factual questions
accurately and promptly from heterogeneous sources already present within an organization,
where “promptly” means users do not have to wait extended periods to reach a conclusion,
without requiring different technical configurations for each source type. Two specific limitations
define this gap: first, existing RAG systems are designed for a single source type and cannot
transparently handle heterogeneous data; second conventional pre-indexed systems do not
reflect real-time content changes without manual re-indexing.
The Retrieval-Augmented Generation (RAG) approach introduced by Lewis et al. (2020)
opens an opportunity to address these limitations by combining retrieval from external sources
with LLM generation capabilities, enabling the system to provide factual, grounded answers
based on actual documents. Izacard and Grave (2021) extend this approach for passage retrieval
in open domains settings dense effectiveness was demonstrated by Karpukhin et al. (2020);
integration of RAG with knowledge graphs for multi-hop reasoning was demonstrated by
Yasunaga et al. (2021); while FAISS has evolved as a scalable vector search infrastructure from
Johnson et al. (2019) to the current library architecture described by Douze et al. (2024), and
Reimers and Gurevych (2019) provided multilingual semantic representations through Sentence-
BERT. However, existing RAG implementations are generally single-source and require different
configurations for each data source type, thereby increasing the technical burden for
organizations with heterogeneous data ecosystems. Evaluating RAG-based QA systems requires
a dual perspective: retrieval quality using Precision@K and MRR (Voorhees 1999), generation
quality using ROUGE-L (Lin 2004) and BLEU-1 (Papineni et al. 2002), and RAG-specific
dimensions proposed by Es et al. (2023), namely faithfulness and answer relevance.
These limitations motivate the present study. This study develops a RAG-based QA
system that is agnostic to data sources, capable of handling unstructured documents (PDF, TXT)
and relational databases (PostgreSQL) in a unified manner through a single configuration
interface (Adapter Pattern), thereby providing a knowledge transfer infrastructure that can be
directly implemented in document-based service organizations. The contributions of this study
lie in three dimensions:
In summary, the contributions of this paper are summarized as follows :

Propose an integrated source-agnostic RAG architecture model that is layered-based
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
and works under various source configurations and real-time conditions.
Implement a layered-based of structural and unstructured knowledge paradigm with
progressive scenarios that isolate and quantify the incremental contribution of each
layer.
Evaluate the model using metrics composite evaluation metrics purpose-build for
multi-source RAG in organizational knowledge transfer contexts: Knowledge
Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS), and Answer
Quality Index (AQI)
RESEARCH METHODS
Data Sources and Preprocessing
This study employs real operational data from a domain-specific financial instrument
transaction platform (BOND_SYS) organized in a three-layer corpus: (L1) BOND_SYS system
specifications in PDF/TXT document format, covering system module descriptions, business
process flows, and technical requirements; (L2) a BOND_SYS PostgreSQL database with 8
operational tables containing real data (20 RFQs, 10 Securities, 10 firms, 10 quotations, 10 trades,
10 trade statuses, 11 firm default parameters, 8 fraction masters); and (L3) developer team
discussion logs in TXT format comprising 908 messages from three sources: a 2022 group
discussion, a February 2025 group discussion and a 2022 personal conversation. System,
institutional and individual identities are anonymized using a token masking scheme (system
name: BOND_SYS, ministry name: GOV_DEPT1, module name: BOND_MOD, etc).
Preprocessing is performed by two adapters according to source type.
FolderSourceAdapter handles document files using the appropriate library per format (pypdf for
PDF, built-in for text). All extracted text is normalized into RawDocument objects with content,
source and format metadata attributes.
PostgreSQLAdapter handles relational sources in three modes: (1) all tables, (2) specific
tables via a table name list (pg_tables), and (3) custom SQL queries (pg_queries). Each table or
query result is converted into structured text that includes column names, row counts, and
tabular data, enabling embedding and retrieval by FAISS. In this study, mode (2) is used with a
predefined list of 8 BOND_SYS operational tables; modes (1) and (3) are supported by the adapter
interface but were not invoked in this evaluation, as the relevant tables were known in advance
and no cross-table aggregation queries were required.
Following extraction, text is segmented using UniversalTextSplitter based on
RecursiveCharacterTextSplitter with chunk_size = 2000 and overlap = 300. The 300 character
overlap is designed to preserve inter-chunk context so that information split at chunk boundaries
is not entirely lost. The larger chunk_size (2000) is selected to support Indonesian-language
financial documents that typically contain long sentences and multi-row tables. This chucking
choice is aligned with retrieval-augmented pretraining evidence that context segmentation and

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
retrieval granularity substantially affect downstream generation quality and factual grounding
(Borgeaud et al. 2022; Shi et al. 2023) Table 1 summarises the file formats supported by
FolderSourceAdapter.

Table 1. File formats supported by FolderSourceAdapter
Format Library Processing
.pdf (^) pypdf Text
extraction
from all
pages
.txt, .md, .log (^) built-in Raw text
The selection of retrieval hyperparameters is justified as follows. The top-K parameter is
set to 8, balancing between recall (more chunks = more potential evidence) and precision (fewer
chunks = less noise in the LLM context window). Empirical testing showed that K < 5 frequently
missed relevant chunks in multi-source scenarios, while K > 10 introduced irrelevant context that
reduced Answer Faithfulness. The similarity threshold of 0.25 (cosine distance) was calibrated
through iterative testing on development queries: thresholds below 0.20 admitted chunks with
minimal semantic relevance, while thresholds above 0.35 excluded moderately relevant chunks
critical for Answer Completeness. The chunk_size of 2000 characters accommodates the structure
of Indonesian-language financial documents, which typically contain compound sentences
averaging 40-60 words and multi-row configuration tables that would be fragmented at smaller
chunk sizes (e.g., 500 or 1000 characters). The overlap of 300 characters (15% of chunk_size)
ensures that sentences spanning chunk boundaries maintain sufficient context for coherent
retrieval.

Evaluation Dataset
The evaluation covers five scenarios, each representing a different source type and
knowledge transfer dimension. All prompts and system responses are derived from real
operational data in Indonesian, anonymized from the BOND_SYS platform. To preserve
linguistic fidelity, prompts shown in the figures and tables are presented in their original
Indonesian, with English translations provided in italics for accessibility. Result excerpts,
however, are presented in English to maintain consistency with the analytical discussion
throughout the paper.
Scenario A, Chat Only (L3, tacit to operational): Developer team discussion logs from

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
BOND_SYS (908 messages, 3 TXT files). Five questions cover operational issues found exclusively
in the discussion logs: submit quotation bug, upload allocation demo issue, a filter-status bug
tracked as NEWCORE-2442 (an internal issue tracker ticket referencing a backend detect in the
WAITING-status filter), decimal digit decision, and amend feature status.
Scenario B, PDF Only (L1, explicit to actionable): BOND_SYS system specification
documents in PDF/TXT format. Five questions cover documented business process flows:
INSTRUMENT_TYPE_A stages, differences between General/Restricted sessions, parties
involved in RFQ approval, Upload Allocation technical requirements, and broadcast notification
mechanism.
Scenario C, PostgreSQL Only (L2, explicit to structured): BOND_SYS PostgreSQL
database with 8 operational tables. Five questions cover structural configuration data: default
price percentage values, fraction_type/digit combinations, auction_unit differences per board,
firms with is_active = Y, and quotations allocated to a specific RFQ.
Scenario D, PDF + DB (L1+L2, explicit to cross-referenced): Combined PDF documents
and PostgreSQL tables (8 tables) through MultiSourceAdapter. Five questions require cross-
referencing specification documents with actual data: session time consistency, board type in
documents vs. DB, offering_parameter consistency, settlement_date calculation, and
offering_digit consistency with fraction_masters.
Scenario E, Hybrid All (L1+L2+L3, Cross-Paradigm): All layers (PDF, PostgreSQL, and
discussion logs) are combined into a single FAISS index. Five questions require all three layers
simultaneously: submit quotation BOARD_TYPE_A bug (Chat+PDF), offering digit decision
(Chat+DB), BOND_SYS ETL incident (anonymized period) (Chat), upload allocation flow
(PDF+DB+Chat), and amend feature status (PDF+DB+Chat). This is the only scenario equipped
with 5 manually curated reference answers (GROUND_TRUTH_HYBRID), making ROUGE-L
and BLEU-1 reference-based, in contrast to Scenarios A-D which are reference-free (vs. retrieved
context).

System Architecture
The system is built using the Adapter Pattern (Gemma E et al., 1994) to enable data source
extensibility without modifying the core pipeline. The architecture consists of seven main
components arranged sequentially: SourceDetector detects the source type from the input
parameter, SourceFactory initializes the appropriate adapter (FolderSourceAdapter or
PostgreSQLAdapter), UniversalTextSplitter segments the extracted text, RuntimeIndexBuilder
constructs the FAISS index in-memory, QueryProcessor performs similarity search on the index,
and AnswerGenerator produces answers along with evaluation of 8 metrics simultaneously. The
overall architecture is illustrated in Figure 1.

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Figure 1. Architecture of the Agnostic Multi-Source RAG System
Inputs (user question, source parameter, Config) enter the pipeline through
SourceDetector, which classifies the source string and delegates document loading to the
corresponding adapter via SourceFactory: FolderSourceAdapter for local files
(PDF/TXT/MD/LOG), PostgreSQLAdapter for relational tables (SELECT * LIMIT 1000,
serialized to text), or MultiSourceAdapter for hybrid sources. Raw documents are split by
UniversalTextSplitter into 2,000-character chunks (300 character overlap), encoded by a singleton
multilingual MiniLM-L12-V2 model (384 dimensions), and indexed in a FAISS in memory vector
store with session-level caching. At query time, QueryProcessor embeds the question, retrieves
top-8 chunks (similarity threshold 0.2), and passes the formatted context to AnswerGenerator,
which calls Gemini 2.5-flash with zero-shot prompting and an automatic retry-and-model-
fallback chain (attempt 2 falls back to gemini-2.0-flash, attempt 4 to gemini 1.5 flash; exponential
backoff with base delay = 15 s, max 6 entries. The final RAG Results is simultaneously evaluated
on 8 quantitative metrics: Retrieval Relevance (RR), Answer Faithfulness (AF), Answer

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Completeness (AC), ROUGE-L, BLEU-1, Precision@K, MRR, and Context Coverage (CC).
Directed edges represent data flow; dashed edges indicate parallel scoring path in which
intermediate retrieval artifacts (chunks and relevance scores) are forwarded to the Evaluator
independently of the generation step.
SourceDetector applies pattern matching rules to automatically detect the source type, as
summarised in Table 2.

Table 2. Automatic detection rules for data source type
Input Pattern Adapter Example
postgresql:// or
postgres://
PostgreSQLAdapter postgresql://user:pass@host/db
Absolute path(/, C:\, ~) FolderSourceAdapter /content/drive/Mydrive/data
Path relatif (./, ../) FoldersourceAdapter ./documents/laporan
Default fallback FolderSourceAdapter Nama folder tanpa prefix
The embedding component uses sentence-transformers/paraphrase-multilingual-
MiniLM-L12-v2 (384 dimensions, CPU, normalize_embeddings=True) as a singleton loaded once
and reused throughout the pipeline. This model supports over 50 languages including
Indonesian. The generative component uses Gemini 2.5-flash as the primary model with
google/flan-t5-base as a fallback.

Model Training and Evaluation
The RAG system does not undergo training (fine-tuning) phase, as it employs pre-trained
models covering multilingual domains. The research focus is on end-to-end pipeline performance
evaluation using eight quantitative metrics.
The FAISS index is constructed in-memory (real-time indexing) at each invocation of
pipeline.ask(), unlike conventional RAG systems that persist the index to disk. The execution
sequence is: (1) the adapter loads documents from the source on-the-fly, (2) the splitter segments
the text, (3) FAISS builds the index in RAM, (4) the query processor performs retrieval, and (5)
the answer generator produces the answer. Session cache (use_session_cache=True) enables
index reuse for the same source within a single execution session, reducing overhead without
sacrificing inter-call data consistency within the same session. This design choice follows the
practical FAISS implementation direction described in recent library documentation (Douze et al.
2024).

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Interpretive Analysis and Evaluation Design
The 8-metric evaluation framework is grouped into three dimensions to enable separate
interpretation of retrieval quality and generation quality:
Retrieval Dimension (Classical IR): Precision@K Eq. (1) measures the proportion of top-
K retrieved chunks that are relevant. MRR Eq. (2) captures rank quality by rewarding systems
that place the first relevant chunk higher. Context Coverage Eq. (3) measures source diversity
across top-K chunks.

Pr𝑒𝑐𝑖𝑠𝑖𝑜𝑛𝐾 =
|𝑅𝑒𝑙𝑒𝑣𝑎𝑛 𝐶ℎ𝑢𝑛𝑘𝑠|
𝐾
(1)^
𝑀𝑅𝑅 =
1
𝑟𝑎𝑛𝑘 𝑜𝑓 𝑓𝑖𝑟𝑠𝑡 𝑟𝑒𝑙𝑒𝑣𝑎𝑛𝑡 𝑐ℎ𝑢𝑛𝑘^
(2)
𝐶𝑜𝑛𝑡𝑒𝑥𝑡𝐶𝑜𝑣𝑒𝑟𝑎𝑔𝑒 =
𝑈𝑛𝑖𝑞𝑢𝑒 𝑆𝑜𝑢𝑟𝑐𝑒
𝑇𝑜𝑡𝑎𝑙 𝐶ℎ𝑢𝑛𝑘𝑠^
(3)
Answer Quality Dimension:
Retrieval Relevance = cosine similarity of query embedding vs. mean of top-K chunk
embeddings
Answer Faithfulness = F1 token overlap of answer vs. combined context (anti-
hallucination)
Answer Completeness = ratio of question keywords present in the answer
NLP Dimension:
ROUGE-L (Lin 2004) = F1 based on Longest Common Subsequence
BLEU-1 (Papineni et al. 2002) = Unigram precision with brevity penalty
In addition to these eight metrics, three composite metrics are defined to address
evaluation questions that cannot be answered by any single metric:
Composite Metric 1, Knowledge Transfer Effectiveness (KTE)
𝐾𝑇𝐸 =
𝐴𝑛𝑠𝑤𝑒𝑟 𝐹𝑎𝑖𝑡ℎ𝑓𝑢𝑙ln𝑒𝑠𝑠 + 𝐴𝑛𝑠𝑤𝑒𝑟 𝐶𝑜𝑚𝑝𝑙𝑒𝑡𝑒𝑛𝑒𝑠𝑠
2
( 4 )
KTE is the average of two components Eq. (4): Faithfulness (proportion of the answer
supported by context, as an anti-hallucination measure) and Completeness (proportion of
question keywords covered in the answer). The effective threshold is set at KTE 0.5. Knowledge
transfer succeeds only if the answer is simultaneously non-hallucinatory and complete. If either

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
component is zero, knowledge transfer fails; the simple average is therefore sufficient as an
effectiveness measure (Nonaka and Takeuchi 1995).
Composite Metric 2, Multi-Source Retrieval Score (MSRS):

𝑀𝑆𝑅𝑆 =
Pr𝑒𝑐𝑖𝑠𝑖𝑜𝑛@𝐾 + 𝐶𝑜𝑛𝑡𝑒𝑥𝑡 𝐶𝑜𝑣𝑒𝑟𝑎𝑔𝑒
2
( 5 )
MSRS is the average of two components Eq. (5): Precision@K (proportion of relevant
chunks in top-K retrieval) and Context Coverage (source document diversity in top-K). A system
that retrieves only from a single file will obtain a low Context Coverage even with a high
Precision@K. MSRS detects this condition and ensures that the multi-source claim is evidenced
at the retrieval level. This approach applies the diversity principle in Information Retrieval
(Carbonell and Goldstein 1998) to the multi-source RAG context.
Composite Metric 3, Answer Quality Index (AQI):

𝐴𝑄𝐼 =
𝐹𝑎𝑖𝑡𝑓𝑢𝑙ln𝑒𝑠𝑠 + 𝐶𝑜𝑚𝑝𝑙𝑒𝑡𝑒𝑛𝑒𝑠𝑠 +𝑅𝑂𝑈𝐺𝐸−𝐿
3
( 6 )
AQI is the average of three components Eq. (6): Faithfulness (anti-hallucination),
Completeness (question coverage), and ROUGE-L (structural similarity based on word sequence
order against context). While KTE only measures whether knowledge is conveyed in terms of
content, AQI adds a linguistic dimension to detect answers that are thematically adequate but
structurally divergent from the source documents. This multi-aspect approach is consistent with
the SummEval methodology (Fabbri et al. 2021).
The relationships among composite metrics: KTE measures from the user perspective
(whether knowledge is conveyed), MSRS from the system perspective (whether multi-source is
evidenced), and AQI from the linguistic perspective (whether the answer is of NLP quality). The
three are complementary; a well-performing system should achieve high scores across all three
dimensions simultaneously.
The aggregate Overall metric is calculated as the simple average of five answer quality
metrics:

𝑜 =
𝑅𝑅 + 𝐹𝑎𝑖𝑡ℎ𝑓𝑢𝑙ln𝑒𝑠𝑠 + 𝐶𝑜𝑚𝑝𝑙𝑒𝑡𝑒𝑛𝑒𝑠𝑠 +𝑅𝑂𝑈𝐺𝐸−𝐿 + 𝐵𝐿𝐸𝑈− 1
5
(6)
Pure retrieval metrics (Precision@K, MRR, Context Coverage) are excluded from Overall
Eq. (7) so that this aggregate reflects answer quality rather than retrieval quality, which is
consistently perfect (P@K = MRR = 1.000).
This three-dimensional evaluation separation enables identification of whether low
values are caused by retrieval failure or by limitations in the generative model’s language
capability, two problems that require fundamentally different solutions.

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
To substantiate the “Multi-Source” claim, the evaluation is designed in five scenarios, each
representing one dimension of organizational knowledge transfer. Scenario E specifically proves
the source-agnostic claim at the highest level: a single query spans PDF/TXT system specification
documents, team discussion logs (TXT), and PostgreSQL tables (SQL) simultaneously within a
single FAISS index. Scenario E is also the only scenario equipped with ground truth, making
ROUGE-L and BLUE-1 reference-based. Table 3 summarises the five scenarios with their
respective adapters and knowledge transfer dimensions

Table 3. Multi-Source evaluation design and knowledge transfer dimensions
Scenario Layer Adapter Source Format KT
Dimension
Knowledge
Type
A: Chat L3 FolderSourceAdapter^ BOND_SYS
discussion
log (
messages, 3
TXT file )
TXT Tacit to
Operational
Percakapan
informal ⇒
jawaban
operasional
B: PDF L1 FolderSourceAdapter BOND_SYS
system spec
( PDF docs)
PDF./TXT Explicit to
Actionable
Format
specification to
business
process insight
C:
PostgreSQL
L2 PostgreSQLAdapter BOND_SYS
DB (8 tables,
20 RFQ)
SQL Explicit to
Structured
Table data to
contextual
narrative
D: PDF +
DB
L1 +
L
MultiSourceAdapter PDF +
PostgreSQL
(combined)
TXT +
SQL
Explicit to
Cross-
referenced
Cross
verification of
specification
and actual data
E: Hybrid L1 +
L2 +
L
MultiSourceAdapter PDF + DB +
Chat (all
layers)
TXT +
SQL
Cross-
Paradigm
Tacit, explicit,
and structured
combined
Scenario E: reference-based evaluation (ROUGE-L and BLEU- 1 vs.
GROUND_TRUTH_HYBRID). Scenarios A-D: reference-free (vs. retrieved context).

RESULT AND DISCUSSION
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
System Performance
1. Multi Source
An initial evaluation was performed on the question “What did the team discuss
regarding the NEWCORE-2442 issue and what is the resolution status based on the conversation
log?” using the BOND_SYS developer team discussion logs (908 messages, 3 TXT files) as the
source, with Gemini 2.5-flash (Google, multilingual) as the generative model. This question
represents a realistic knowledge retrieval scenario in a software development environment,
where users need to identify discussion related to a specific issue and determine its resolution
status without manually reviewing hundreds of messages. Evaluation results are reported in
Table 4.

Table 4. Single-query evaluation results (question A3) using Gemini 2.5-flash
Metric Gemini 2.5 Flash Dimension
Retrieval Relevance 0.460 Retrieval
Answer Faithfulness 0.064 Generation
Answer
Completeness
0.833 Generation
ROUGE-L 0.029 Generation
BLEU- 1 0.000 Generation
Precision@K 1.000 Retrieval
MRR 1.000 Retrieval
Overall 0.277 Combined
The retrieval component achieved strong ranking performance, as indicated by perfect
Precision@K(1.000) and MRR (1.000) scores, demonstrating that the relevant context was
successfully retrieved and ranked at the top position. However, the Retrieval Relevance score of
0.460 suggests that the retrieved documents were only moderately aligned with the query
semantics.
On the generation side, the system achieved high Answer Completeness (0.833),
indicating that the generated response covered most of the information required to answer the
question. Nevertheless, the low Answer Faithfulness score (0.064), together with low ROUGE-L

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
(0.029) and BLEU-1 (0.000) values, suggests that the response was highly abstractive and did not
closely match the wording of the source documents or reference answer. This behaviour is
expected when using large language models to summarize conversational data, particularly in
multilingual and informal communication contexts where semantically correct answers may
differ substantially from the original phrasing.

2. Multi-Source Batch Evaluation
To substantiate the Multi-Source claim and measure knowledge transfer effectiveness, a
batch evaluation was conducted across five scenarios with 25 questions in total (5 per scenario).
Scenarios A, B, and C evaluate each corpus layer separately. Scenario D evaluates the L1 + L
combination and Scenario E evaluates all three layers simultaneously through
MultiSourceAdapter. Scenario E is the only one that employs reference-based evaluation (against
5 manually curated reference answers), while Scenarios A-D use reference-free evaluation
(against retrieved context). All results are produced below using the Gemini 2.5-flash model.
Table 5a. Retrieval summary per scenario

Scenario Adapter Format N RR P@K MRR CC
A: Chat
Only (L3)
FolderSourceAdapter TXT 5 0.473 1.000 1.000 0.
B: PDF (L1) FolderSourceAdapter^ TXT 5 0.580 1.000 1.000 0.
C:
PostgresQL
(L2)
PostgresQLAdapter (^) SQL 5 0.460 1.000 1.000 0.
D: PDF +
DB (L1+L2)
MultiSourceAdapter (^) TXT + SQL 5 0.576 1.000 1.000 0.
E: Hybrid
(L1+L2+L3)
MultiSourceAdapter TXT + SQL 5 0.582 1.000 1.000 0.
RR = Retrieval Relevance; CC = Context Coverage; P@K dan MRR = 1.000 across all scenarios.
Table 5b. Answer quality and composite metrics per scenario
Scenario Faith Comp ROUGE-
L

BLEU-
1
Overall KTE MSRS AQI
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
A: Chat
(L3)
0.075 0.853 0.026 0.000 0.285 0.464 0.725 0.
B: PDF (L1) 0.114 0.675 0.046 0.000 0.284 0.395 0.625 0.
C:
PostgreSQL
(L2)
0.0.44 0.581 0.022 0.000 0.221 0.312 0.825 0.
D: PDF+DB
(L1+L2)
0.117 0.661 0.041 0.000 0.279 0.389 0.762 0.
E: Hybrid
(L1+L2+L3)
0.127 0.779 0.181 0.196 0.373 0.453 0.713 0.
F aith = Answer Faithfulness; Comp = Answer Completeness.
KTE = (Faith + Comp) / 2. MSRS = (P@K + CC ) / 2. AQI = (Faith + Comp + ROUGE-L) / 3.
Overall = (RR + Faith + Comp + ROUGE-L + BLEU-1) / 5
Skenario E: ROUGE-L dan BLEU-1 reference-based. Scenarios A-D: reference-free.
All values rounded to three decimal places; rounding differences of 0.001 may occur in composite
formulas.
To substantiate the Multi-Source claim and measure knowledge transfer effectiveness, a
batch evaluation was conducted across five scenarios with 25 questions in total (5 per scenario).
Scenarios A, B, and C evaluate each corpus layer separately. Scenario D evaluates the L1 + L
combination and Scenario E evaluates all three layers simultaneously through
MultiSourceAdapter. Scenario E is the only one that employs reference-based evaluation (against
5 manually curated reference answers), while Scenarios A-D use reference-free evaluation
(against retrieved context). All results are produced below using the Gemini 2.5-flash model.
3. Ablation Study: Layer Contribution Analysis
To demonstrate that each corpus layer provides a measurable contribution, an ablation
study was conducted on the same questions (E1-E5) across four progressively expanded source
configurations.
Table 6. Ablation study results (per-layer contribution)
Configuration Active
Layer

n Overall Faithfulness Completeness ROUGE-
L
BLEU-
1
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Ablation-0:
Chat

L3 5 0.237 0.076 0.636 0.026 0.
Ablation-1:
PDF

L1 5 0.214 0.139 0.320 0.053 0.
Ablation-2:
PDF + DB

L1+L2 5 0.234 0.128 0.428 0.051 0.
Full: PDF +
DB + Chat

L1+L2+L3 5 0.373 0.127 0.779 0.181 0.
Ablation-0 through Ablation-2: reference-free (vs. retrieved context). Full (L1+L2+L3):
reference-based (vs. GROUND_TRUTH_HYBRID). Overall = (RR + Faithfulness +
Completeness + ROUGE-L + BLEU-1) / 5; the RR column is not displayed as it is collected
alongside other metrics. Ablation-Full is identical to Scenario E in Tables 5a/5b (RR = 0.582).
The ablation results show a clear contribution from combining all three knowledge layers.
The Overall score remains relatively similar across Chat-Only (0.237), PDF-only (0.214), and
PDF+db (0.234), but increases substantially to 0.373 when the complete three-layer configuration
(L1+L2+L3) is used, representing a gain of +0.139 over PDF+DB. Similar improvements are
observed in ROUGE-L, which increases from 0.051 to 0.181 and BLEU-1, which rises from 0.
to 0.196 after Layer 3 (Chat) is integrated.
These findings indicate that conversational knowledge contained in team discussion logs
provides information that is not fully represented in formal documentation or structured
databases. The improvement is particularly evident for operational and historical questions
requiring contextual knowledge, including quotation issues, offering digit decisions, ETL
incident investigations, allocation workflows and feature implementation status. Therefore, the
result suggests that the effectiveness of the proposed system is derived not only from the multi-
adapter architecture but also from the integration of complementary knowledge layers within the
corpus.
4. Running Interface example and Prompt-Level Results
To complement aggregate quantitative metrics, this subsection presents one running-
interface example and a prompt-level summary table. The running interface screenshot is taken
from Scenario A (Chat Only), showing one complete interaction consisting of a user prompt and
system response in the original operational language.
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Figure 2. Running-interface example (Scenario A): prompt and response are shown in
original Indonesian.

Table 8 summarises five prompt-level examples from Scenario A (Chat Only). Prompt and
result texts are presented in Indonesian (original), while performance metrics are reported in
English.

Table 8. Prompt-level results (scenario A: chat only)
No. Prompting Result Performance
1 Masalah submit
quotation board
BOARD_TYPE_A
menjelang demo dan
workaround
sementara?
Problem submitting
quotation board
BOARD TYPE A before
demo and temporary
workaround?
The answer did not find sufficient contextual
evidence to explain the details of the
problem and the workaround.
RR=0.441;
P@K=1.000;
MRR=1.000;
Overall=0.269;
KTE=0.
2 Mengapa upload
allocation tidak dapat
didemonstrasikan
pada sesi demo 27 Juli
2022
The answer identifies, based on the team's
discussion log on July 27, 2022, that the
upload allocation feature could not be
demonstrated because of a feature update
RR=0.516;
P@K=1.000;
MRR=1.000;
Overall=0.282;
KTE=0.
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)

Why couldn't upload
allocation be
demonstrated during the
demo session on July 27,
2022?
from DEV_E, which rendered the file
unusable for upload allocation.
3 Apa yang
didiskusikan tim
terkait isu
NEWCORE-2442 dan
apa status
penyelesaiannya
berdasarkan log
percakapan?
What did the team
discuss regarding the
NEWCORE-2442 issue,
and what is its
resolution status based
on the conversation log?
The answer captures the following.
The team discussed the NEWCORE- 2442
issue regarding:
A backend gap in filtering the
WAITING status and outputting the
CLOSED and ALLOCATED statuses.
A question on whether the feature
needs to operate in real time.
Based on the conversation log, the resolution
status is as follows.
Tim mendiskusikan isu NEWCORE- 2442
terkait dengan:
The backend still requires support for
filtering the WAITING, CLOSED, and
ALLOCATED statuses.
The feature is not yet real-time.
There was a discussion on adding a
WAITING status to the database and
the backend requirements needed for
it.
RR=0.460;
P@K=1.000;
MRR=1.000;
Overall=0.277;
KTE=0.
4 Apa keputusan teknis
yang didiskusikan tim
terkait jumlah desimal
(digit), untuk last
price dan offering
price?
What technical decision
did the team discuss
regarding the number of
The answer identifies the following,
Based on the given context, the technical
decision discussed by the team regarding the
number of decimal digits for last price and
offering price is as follows:
RR=0.473;
P@K=1.000;
MRR=1.000;
Overall=0.304;
KTE=0.
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)

decimal digits for last
price and offering price?
Last Price and Offering Price must be
able to accept and store up to 5 decimal
digits.
Input values (e.g., 30.12345) must be
stored exactly as entered (30.12345),
rather than being rounded or truncated
to 2 decimal digits (e.g., 30.12).
This change will not be affected by the
offering parameter.
Previously, there was a table-level validation
limiting input to only two decimal digits,
which DEV_B referred to as a "hardcode."
However, after confirmation from PO_1 with
the user, it was agreed to change this to
accommodate the user's requirements.
5 Apa status
implementasi fitur
amend pada modul
trade custody
berdasarkan diskusi
tim Februari 2025?
What is the
implementation status
of the amend feature in
the trade custody
module, based on the
team's February 2025
discussions?
The answer identifies the following,
Based on the team's discussions in February
2025, the amend feature in the trade custody
module is currently in the development and
testing stage.
Several points regarding its implementation
status are as follows:
There was a discussion on the "confirm
amend" behavior in trade custody, where the
second confirmation (confirm amend) does
not create a new row or convert the previous
one to inactive, unlike the first confirmation.
(09/02/2025)
PO_3 stated that the value in the amend
should be recalculated if there is a change.
(12/02/2025)
Ini This indicates that the feature is still
RR=0.474;
P@K=1.000;
MRR=1.000;
Overall=0.293;
KTE=0.
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
undergoing development, testing, and
refinement, with several functional and
display issues currently being addressed.
5. Cross-Scenario Prompt Level Highlights
To provide granular insight beyond scenario-level means, Table 9 presents representative
prompt-level results from Scenarios B-E, selecting the highest-scoring and lowest-scoring
question per scenario to illustrate performance variance. Several observations emerge from the
prompt-level analysis:

Table 9. Selected prompt-level results across scenarios (best and worst per scenario)
Scenario No. Prompt RR Faith Comp KTE Overall
B (Best) B2 Apa perbedaan antara sesi
General dan sesi Restricted
dalam sisstem BOND_SYS
?
What is the difference
between the General session
and the Restricted session in
the BOND_SYS system?
0.526 0.179 0.875 0.527 0.
B (Worst) B5 Bagaimana mekanisme
pengiriman notifikasi
broadcast kepada peserta
setelah alokasi?
What is the mechanism for
sending broadcast
notifications to participants
after allocation?
0.613 0.104 0.375 0.240 0.
C (Best) C5 Dari semua quotation
pada RFQ 20140327-01,
berapa yang dialokasikan
?
0.565 0.030 0.857 0.444 0.
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)

Of all the quotations under
RFQ 20140327-01, how
many were allocated?
C (Worst) C3 Apa perbedaan
auction_unit antara board
BS-SB dibandingkan board
lainnya?
What is the difference in
auction_unit between the BS-
SB board and other boards?
0.277 0.050 0.167 0.109 0.
D (Best) D4 Berdasarkan FR dan data
aktual, bagaimana
settlement_date ditetapkan
?
Based on the FR and actual
data, how is the
settlement_date determined?
0.676 0.185 0.857 0.521 0.
D (Worst) D1 Apakah jam sesi pada data
RFQ aktual sudah sesuai
spesifikasi?
Does the session time in the
actual RFQ data comply with
the specification?
0.534 0.121 0.200 0.161 0.
E (Best) E2 Apa keputusan teknis
terkait offering digit dan
validasi hardcode yang
didiskusikan oleh tim?
What technical decision
regarding the offering digit
and hardcode validation was
discussed by the team?
0.588 0.112 0.900 0.506 0.
E (Worst) E3 Berdasarkan log diskusi
tim, apa yang terjadi pada
insiden ETL BOND_SYS?
0.491 0.047 0.667 0.357 0.
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Based on the team's
discussion log, what
happened during the
BOND_SYS ETL incident?
6. Error Analysis and Failure Cases
A systematic examination of failure cases reveals three distinct failure patterns in the
system's responses:
Pattern 1: Complete Retrieval Failure ("Information Not Found"). Question A1 ("Apa
masalah yang ditemukan tim saat proses submit quotation pada board BS-SB menjelang demo,
dan bagaimana workaround sementara yang disepakati?") produced a response explicitly
stating: "Informasi tidak ditemukan dalam sumber data yang tersedia. Konteks tidak
menyebutkan masalah yang ditemukan tim saat proses submit quotation pada board BS-SB
menjelang demo." Despite achieving P@K = 1.000 and RR = 0.441, the generative model
determined that the retrieved chunks did not contain sufficient evidence to answer the question.
This represents a conservative behaviour where the model refuses to hallucinate rather than
generating an unsupported answer. Notably, Answer Completeness remained high (0.867)
because the response echoes question keywords, while Answer Faithfulness was very low (0.031)
due to the absence of substantive content grounded in the context.
Pattern 2: Low Completeness from Structured Data. Question C3 achieved the lowest
Overall score across all 25 questions (0.103) with Answer Completeness = 0.167. The system
correctly retrieved and reported the auction_unit values per board (BS-SB = "Mio", BS = "Bio", BC
= "Bio"), but the answer was concise and factual rather than explanatory. Token-overlap metrics
penalize short, correct answers because they lack the verbose explanation needed to cover
question keywords like "perbedaan" (difference). This represents a metric limitation rather than
a system failure: the answer is factually correct but scores poorly on Completeness because the
evaluation metric rewards keyword coverage over factual accuracy.
Pattern 3: Partial Answer with Cross-Reference Gap. Question D1 (Overall = 0.178,
Completeness = 0.200) required comparing session times from FR specification documents
against actual RFQ data in the database. The system retrieved relevant chunks from both sources
but generated only a partial answer covering the specification side without explicitly cross-
referencing the database values. This reveals a limitation in the generative model's ability to
perform explicit comparison across heterogeneous source types within a single answer, even
when relevant contexts from both sources are successfully retrieved.
Summary of failure distribution: Of 25 questions evaluated, 1 produced a complete
retrieval failure (Pattern 1), 3 achieved Overall < 0.200 due to low Completeness from

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
structured/comparative queries (Pattern 2), and 2 exhibited partial cross-reference gaps (Pattern
3). The remaining 19 questions (76%) achieved Overall ≥ 0.228 with substantive answers covering
the majority of question intent. These patterns suggest that future improvements should
prioritize: (a) enhanced prompt engineering for comparative reasoning, (b) answer expansion
strategies for structured data queries, and (c) explicit cross-source synthesis instructions in the
generation prompt.

Visualization of Results
Evaluation results are visualized across four separate panels (Figures 3a-3d):
Figure 3a presents the mean standard metrics per scenario in a grouped bar chart. P@K
and MRR consistently reach 1.000 across all scenarios, indicating that the retrieval component
performs optimally regardless of source type configuration. This ceiling effect in retrieval metrics
confirms that the combination of multilingual MiniLM-L12-v2 embeddings and FAISS
IndexFlatL2 provides sufficient discriminative power for the 86-chunk corpus. Scenario B exhibits
the highest ROUGE-L among reference-free scenarios (0.051), while Scenario E achieves ROUGE-
L = 0.181 and BLEU-1 = 0.196 (reference-based). The substantial gap between reference-free and
reference-based ROUGE-L values (0.051 vs. 0.181) suggests that reference-free ROUGE-L
systematically underestimates generation quality, as it compares against retrieved chunks rather
than ideal answers.
Figure 3b presents the overall score per question (25 questions across 5 scenarios),
revealing the distribution of performance across questions and scenarios. A notable observation
is the variance within scenarios: Scenario A exhibits relatively consistent scores (range: 0.269–
0.304), while Scenario C shows higher variance (range: 0.165–0.289), indicating that structured
database content is more sensitive to question formulation. Questions requiring numeric lookup
from PostgreSQL tables (e.g., "default price percentage values") achieve lower Overall scores than
questions requiring narrative synthesis, because token overlap metrics (Faithfulness, ROUGE-L)
penalize concise numeric answers that are factually correct but lexically dissimilar from the
context.
Figure 3c compares three composite metrics (KTE, MSRS, AQI) across scenarios. MSRS is
highest in Scenario C (PostgreSQL, 0.825), reflecting that the 8-table corpus provides high source
diversity in top-K retrieval. Conversely, Scenario C records the lowest KTE (0.312), revealing a
trade-off: high source diversity does not guarantee effective knowledge transfer when the source
content is structured data lacking narrative explanation. Scenario A (Chat Only) achieves the
highest KTE (0.464) despite relatively low MSRS (0.725), suggesting that conversational
knowledge, while concentrated in fewer sources, provides richer contextual information for
operational question answering. Scenario E yields KTE = 0.453 despite its questions being the
most complex, demonstrating that multi-source integration compensates for per-layer
limitations.

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Figure 3d is a radar chart illustrating the multi-dimensional profiles of all five scenarios,
confirming that each scenario exhibits a distinct and complementary profile: A leads in
Completeness (0.853) and also leads among reference-free scenarios in Overall (0.285) and KTE
(0.464), indicating conversational data covers question keywords comprehensively and supports
effective transfer; B leads in Faithfulness (0.114) among single-source scenarios, showing formal
documents provide more faithful grounding; C leads in MSRS (0.825) due to source diversity
from 8 tables; and E leads in Retrieval Relevance (0.582) and reference-based metrics. The radar
chart visually confirms that no single scenario dominates all dimensions simultaneously,
validating the multi-scenario evaluation design.

Figure 3a. Mean standard metrics per scenario (grouped bar chart)
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)

Figure 3b. Overall score per question (25 questions, 5 scenarios)
Figure 3c. Composite metrics KTE, MSRS and AQI per scenario
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Figure 3d. Radar chart of multi-dimensional profiles across five scenarios
Result Limitation
The retrieval component uses paraphrase multilingual-MiniLM-L12-v2 (50+ languages),
ensuring no language bias at the retrieval layer; retrieval weights are entirely determined by
semantic similarity, not source type metadata. For single-entity questions, Context Coverage
tends to be low because top-K chunks concentrate on a single document; this accurately reflects
the corpus structure rather than a system bias. Content hash-based deduplication is implemented
to prevent a single file from being counted as multiple distinct sources.
Several limitations of this study must be explicitly acknowledged:

Evaluation coverage. The evaluation covers 25 questions across five scenarios within
a single domain (a domain-specific financial instrument transaction platform,
BOND_SYS) using an Indonesian-language corpus. Empirical validation across other
domains and languages is required to strengthen the generalizability of the results.
Partial ground truth. Only Scenario E (5 questions) is equipped with manually curated
reference answers (ground truth). Scenarios A-D employ reference-free evaluation,
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
such that ROUGE-L and BLEU-1 for those four scenarios are computed against
retrieved context rather than ideal reference answers. Scenario E’s reference-based
ROUGE-L (0.181) and BLEU-1 (0.196) cannot be directly compared with the reference-
free ROUGE-L of Scenarios A-D (0.022-0.046). Extending ground truth to all 25
questions would improve comparative validity across scenarios.
Evaluation scale. Evaluation across 25 questions in five scenarios provides an
adequate proof-of-concept, but is insufficient for statistically generalizable claims.
Future research is recommended to employ at least 20-50 query-answer pairs per
scenario with curated ground truth across all scenarios.
LLM non-determinism. The generative component (Gemini 2.5-flash) is inherently
non-deterministic: identical prompts may produce slightly different answers across
runs, potentially affecting token-overlap metrics (Faithfulness, ROUGE-L, BLEU-1).
All reported results are from a single evaluation run; repeated trials with statistical
aggregation would strengthen the reliability of generation-side metrics.
Composite metric validity. The three composite metrics proposed in this study (KTE,
MSRS, AQI) are author-defined formulations designed for multi-source RAG
evaluation. While each component metric is grounded in established IR and NLP
literature, the specific composite formulations have not been independently validated
by the research community. Their applicability beyond the evaluation context of this
study should be verified through replication in other multi-source RAG settings.
Implication
1. Implication for Organizational Knowledge Transfer
The system directly addresses the challenge identified in the introduction. A newly
onboarded Product Owner can immediately pose questions in natural language, for example,
“What bug was found in the submit quotation on board BOARD_TYPE_A and how was it
resolved?” or “What is the implementation status of the amend feature in the
BOND_MOD_CUSTODY?” , and the system will automatically search for answers from the
combination of PDF system specification documents, 908-message team discussion logs and the
BOND_SYS PostgreSQL database within a single query. This scenario represents knowledge
transfer from previous personnel to their successors without requiring a complete re-reading of
all scattered documentation, consistent with the challenge of tacit knowledge explicitation
identified by Nonaka and Takeuchi (1995) and with knowledge management system design
requirements highlighted by Alavi and Leidner (2001). The same principle applies to other
organizational contexts: new analysts can query technical decisions or system configurations
from existing project documentation.

2. Implication for System Extensibility
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
The Adapter Pattern architecture opens a path for extending to new source types.
MongoDB, SharePoint and Google Drive API can each be integrated by implementing two
methods, load() and describe(), without modifying the core pipeline. This aligns with the
Open/Close Principle in software design (Martin 2017). The real-time vs. pre-indexed trade-off
should be considered according to use case, as summarised in Table 7.

Table 7. Trade-off Comparison: Real-Time vs. Pre-Indexed Indexing
Aspect Real-time (this system) Pre-indexed
Data consistency Always up-to-date Potentially stale
Cold start time ~2-5 seconds Instan
Storage
management
No disk overhead Requires synchronization
Best suited for Frequently changing
documents
Large static corpora
For very large static corpora (millions of documents), FAISS IVF (Inverted File Index) or
hybrid search integration with BM25-based hybrid search can be considered as architectural
evolution (Roberstson and Zaragoza 2009; Manning et al. 2008).

3. Implications for RAG Evaluation Methodology
The evaluation findings of this study reveal several methodological insights applicable to
future RAG system evaluations. First, the consistent Precision@K = 1.000 and MRR = 1.000 across
all 25 questions demonstrates that for small-to-medium corpora (< 100 chunks), FAISS
IndexFlatL2 with appropriate embedding models achieves perfect retrieval, rendering retrieval
metrics uninformative as discriminators. Future evaluations on similar corpus scales should
prioritize generation of quality metrics and composite metrics that capture answer utility rather
than retrieval precision alone.
Second, the divergence between reference-free and reference-based ROUGE-L (0.022–
0.046 vs. 0.181) highlights a fundamental measurement of asymmetry. Reference-free ROUGE-L,
computed against retrieved context, penalizes abstractive generation models that synthesize
information rather than extracting verbatim. This observation suggests that reference-free
evaluation is better suited as a faithfulness proxy (measuring extractive alignment) rather than as
a quality indicator, a distinction that should be explicitly stated in future RAG evaluations lacking
ground truth.

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Third, the composite metrics proposed in this study (KTE, MSRS, AQI) address evaluation
gaps not covered by individual metrics. KTE captures whether knowledge transfer succeeds from
both anti-hallucination and coverage perspectives simultaneously. MSRS prevents single-source
retrieval from being misinterpreted as multi-source capability. AQI bridges content adequacy
with linguistic quality. These composite formulations provide a reusable evaluation template for
other multi-source RAG implementations in organizational contexts, particularly where
evaluation resources are limited and full ground truth annotation is impractical.
Fourth, the partial ground truth strategy employed in this study, where only the most
complex scenario (E) receives curated reference answers, represents a pragmatic evaluation
design for resource-constrained research settings. This strategy concentrates annotation effort on
the scenario most likely to reveal system limitations while using reference-free metrics as lower-
bound estimates for simpler scenarios. Future studies may adopt this graduated approach when
full-corpus annotation is prohibitively expensive.

CONCLUSION
This study successfully develops an agnostic multi-source RAG system fulfilling the
research objectives, with six principal findings:

Agnostic architecture realized: SourceDetector + SourceFactory + BaseSourceAdapter
enable handling of folders (TXT/PDF) and PostgreSQL (3 query modes) from a single
source parameter, without modifications to the core pipeline. Automatic source type
detection via pattern matching operates consistently across all 25 questions.
Real-time indexing confirmed: Each invocation of pipeline.ask() constructs the FAISS
index in-memory with no index files on disk. Content changes in the source are
immediately reflected in retrieval results without system restart.
Multi-source empirically proven: Batch evaluation of 25 questions (5 scenarios x 5
questions) shows Precision@K = 1.000 and MRR = 1.000 across all scenarios. Aggregate
performance: A (Chat, Overall = 0.285, MSRS = 0.725), B (PDF, Overall = 0.284, MSRS
= 0.625), C (PostgreSQL, Overall = 0.221, MSRS = 0.825), D (PDF + DB, Overall = 0.279,
MSRS = 0.763), E (Hybrid, Overall = 0.373, MSRS = 0.713). Scenario E proves the
source-agnostic claim at the highest level: Retrieval Relevance = 0.582, the highest
among all scenarios, confirming that a single query can reach relevant chunks from all
three corpus layers simultaneously.
Ground truth Scenario E yields meaningful ROUGE-L and BLEU-1: With 5 manually
curated reference answers, Scenario E achieves ROUGE-L = 0.181 and BLEU-1 = 0.196
(reference-based), demonstrating genuine content overlap between AI-generated
answers and ideal answers for cross-layer questions. This represents approximately a
~4x improvement over the highest reference-free ROUGE-L (Scenario B, ROUGE-L =
0.046).
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Ablation study proves dramatic Layer 3 contribution: As detailed in 4.1.3, Overall
scores remain relatively similar across Chat-only, PDF-only, and PDF + DB
configurations (0.237, 0.214, 0.234 respectively), but jump sharply to 0.373 when all
three layers are activated (+0.139 over PDF+DB). ROUGE-L increases from 0.051 to
0.181 and BLEU-1 from 0.002 to 0.196, proving that team discussion logs (Layer 3) are
the determining component for cross-paradigm answer quality.
Knowledge transfer effectiveness measured: KTE per scenario: A = 0.464 (tacit to
operational), B = 0.395 (explicit to actionable), C = 0.312 (explicit to structured), D =
0.389 (explicit to cross-referenced), E = 0.453 (cross-paradigm). Scenario A achieves
the highest KTE (0.464), followed by E (0.453) and B (0.395), demonstrating that
conversational knowledge sources and multi-source integration consistently produce
effective knowledge transfer.
The principal contributions of this study are: (i) an Adapter Pattern design that separates
concern among data sources, text processing, retrieval, and generation; (ii) a five-scenario
evaluation design with a three-layer real operational corpus; and (iii) a partial ground truth
framework that enables reference-based validation on priority scenarios without requiring
complete ground truth across all scenarios.
For future research, the following directions are recommended: (1) extending ground
truth to all 25 questions for stronger comparative validation; (2) hybrid search (FAISS + BM25) to
improve retrieval on queries with domain-specific tokens; (3) fine-tuning the embedding model
on organization-specific domain documents; and (4) comparison with commercial vector
database-based RAG systems (Pinecone, Weaviate) as architectural baselines.
REFERENCE
Chui, M, Manyika J, Bughin J, Dobbs R, Roxburgh C, Sarrazin H, Sands G, Westergreen M (2012)
The social economyL unlocking value and productivity through social technologies.
McKinsey Global InstituteFriends and neighbors on the web. Soc Networks 25(3):211–230.
https://www.mckinsey.com/industries/technology-media-and-
telecommunications/our-insights/the-social-economy
Alavi MJ, Leidner DE (2001) Review: knowledge management and knowledge management
system. MIS Q 25(1): 107-136. https://doi.org/10.2307/3250961
Borgeaud S, Mensch A, Hoffman J, Cal T, Rutherford E, Millican K, van den Driessche g, Lespiau
JB, Damoc B, Clark A, et al. (2022) Improving language models by retrieving from trillions
of tokens. In: Proceedings of the 39th International Conference on Machine Learning
(ICML). https://doi.org/10.48550/arXiv.2112.04426
Carbonell J, Goldstein J (1998) The use of MMR, diversity-based reranking for reordering
documents and producing summaries. In: Proceedings of the 21st Annual Annual I

Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
nternational ACM SIGIR Conference on Research and Development in Information
Retrieval, pp 335-336. https://doi.org/10.1145/290941.291025
Es s, James J, Espinosa-Anke L, Schockaert S (2023) RAGAS: automated evaluation of retrieval
augmented generation. arXiv:2309.15217. https://doi.org/10.48550/arXiv.2309.15217
Douze M, Guzhva A, Deng C, Johnson J, Szilvasy G, Mazare PE, Lomeli M, Joulin A, Jegou H
(2024) The FAISS library. arXiv:2401.08281. https://doi.org/10.48550/arXiv.2401.08281
Fabbri AR, Krycinski W, McCann B, Xiong C, Socher R, Radev D (2021) SummEval: re-evaluating
summarization evaluation. Trans Assoc Comput Linguist 9:391-409.
https://doi.org/10.1162/tacl_a_00373
Gao Y, Xiong Y, Gao X, Jia K, Pan J, Bi Y, Dai Y, Sun J, Wang M, Wang H (2024) Retrieval-
augmented generation for large language models: a survey. arXiv:2312.1097.
https://doi.org/10.48550/arXiv.2312.10997
Gamma E, Helm R, Johnson R, Vlissides J (1994) Design patterns: elements of reusable object-
oriented software. Addison-Wesley.
https://books.google.co.id/books?id=6oHuKQe3TjQC&printsec=frontcover&source=g
bs_ge_summary_r&cad=0#v=onepage&q&f=false
Gartner (2020) Gartner says employees spend too much time on low-value tasks: use AI and
automation to fix it. Gartner Newsroom.
https://www.gartner.com/en/newsroom/press-releases/2020- 01 - 23 - gartner-says-
employees-spend-too-much-time-on-low-value-tasks
IDC (2023) 90% of data is unstructured and it’s full of untapped value. IDC Blog.
https://blogs.idc.com/2023/05/09/90-of-data-is-unstructured-and-its-full-of-
untapped-value/
Izacard G, Grave E (2021) Leveraging passage retrieval with generative models for open domain
question answering. In: Proceedings of the 16th Conference of the European Chapter of
the Association for Computational Linguistics, pp 874 - 880.
https://doi.org/10.18653/v1/2021.eacl-main.74
Ji Z, Lee N, Frieske R, Yu T, Su D, Xu Y, Ishil E, Bang YJ, Mandotto A, Fung P (2023) Survey of
hallucination in natural language generation. ACM Comput Surv 55(12):1-38.
https://doi.org/10.1145/3571730
Johnson J, Douze M, Jegou H (2019) Billion-scale similarity search with GPUs. IEEE trans Big Data
7 (3):535-547. https://doi.org/10.1109/TBDATA.2019.2921572
Kapurkhin V, Oguz B, Min S, Lewis P, Wu L, Edunov S, Chen D, Yih W (2020) Dense passage
retrieval for open-domain question answering. In: Proceedings of EMNLP 2020, pp 6769-

https://doi.org/10.18653/v1/2020.emnlp-main.550
Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, Kiela D (2020) Retrieval-Augmented
generation for knowledge-intensive NLP tasks. In: Advances in Neural Information
Processing Systems 33, pp 9459-9474. https://doi.org/10.48550/arXiv.2005.11401
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and
Database Question Answering
(Adi, et al.)
Lin CY (2004) ROUGE: a package for automatic evaluation of summaries. In: Proceedings of the
ACL Workshop on Text Summarization Branches Out. https://aclanthology.org/W04-
1013
Manning CD, Raghavan P, Schutze H (2008) Introduction to information retrieval. Cambridge
University Press, Cambridge.
Martin RC (2017) Clean architecture: a craftsman’s guide to software structure and design.
Prentice Hall, Upper Saddle River
Nonaka I, Takeuchi H (1995) the knowledge-creating company. Oxford University Press, New
York.
Papineni K, Roukos S, Ward T, Zhu WJ (2002) BLEU: a method for automatic evaluation of
machine translation. In: Proceedings of the 40th Annual Meeting of Association for
Computational Linguistics, pp 311-318. https://doi.org/10.3115/1073083.1073135
Robertson S, Zaragoza H (2009) The probabilistic relevance framework: BM25 and beyond. Found
Trends Inf Retr 3(4):333-389. https://doi.org/10.1561/1500000019
Reimers N, Gurevych I (2019) Sentence-BERT: sentence embeddings using Siamese BERT-
networks. IN: Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing. https://doi.org/10.18653/v1/D19- 1410
Ren H, Shi H, Zhao W, Zhao J, Zhao Y (2023) Self-RAG: learning to retrieve, generate, and critique
through self-reflection. arXiv:2307.11019. https://doi.org/10.48550/arXiv.2307.11019
Shi P, Li X, Han X, Chang B, Sui Z (2023) REPLUG: retrieval-augmented black-box language
models. arXic:2301.12652. https://doi.org/10.48550/arXiv.2301.12652.
Voorhees EM (1999) The TREC-8 question answering track report. In: Proceedings of the 8th Text
retrieval Conference (TREC-8), pp 77 - 82.
https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf.
Yasunaga M, Ren H, Bosselut A, Liang P, Leskovec J (2021) QA-GNN: reasoning with language
models and knowledge graphs for question answering. In: Proceedings of the 2021
Conference of the North American Chapter of the association for Computational
Linguistics, pp 535-545. https://doi.org/10.18653/v1/2021.naacl-main.45
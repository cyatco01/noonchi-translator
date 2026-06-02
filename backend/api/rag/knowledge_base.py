"""
Sociolinguistic knowledge base for Korean formality-aware translation.

Each entry is tagged with which formality tokens it applies to.
The retriever uses these tags to filter retrieved notes to only those
relevant to the requested speech level.

applies_to values: "formal", "polite", "casual", or multiple.
"""

ENTRIES = [
    # ── Honorific verb pairs (formal) ────────────────────────────────────────

    {
        "id": "hv_eat",
        "text": (
            "For 'eat' or 'drink' addressed to an honored person, use 드시다 instead of "
            "먹다/마시다. Formal question: 드시겠습니까? Formal statement: 드셨습니다."
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "hv_sleep",
        "text": (
            "For 'sleep' when the subject is the addressee, use the honorific 주무시다 "
            "instead of 자다. Example: 주무셨습니까? (Did you sleep well?)"
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "hv_ask",
        "text": (
            "When the speaker asks/inquires of an honored addressee, use 여쭤보다 or 여쭙다 "
            "instead of 묻다. Example: 여쭤봐도 되겠습니까? (May I ask?)"
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "hv_give",
        "text": (
            "Use 드리다 (honorific) instead of 주다 when giving something to an honored "
            "addressee. Example: 드릴까요? (Shall I give it to you?)"
        ),
        "applies_to": ["formal", "polite"],
    },
    {
        "id": "hv_exist",
        "text": (
            "Use 계시다 instead of 있다 when the honored person is the subject. "
            "Example: 계십니까? (Are you there?) vs. 있어요? (casual polite)"
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "hv_say",
        "text": (
            "Use 말씀하시다 when the honored person is speaking. Use 말씀드리다 when "
            "the speaker humbly addresses the honored person. "
            "Example: 말씀하셨습니까? (Did you say?)"
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "hv_know",
        "text": (
            "Use 아시다 instead of 알다 when the honored person is the subject who knows. "
            "Example: 아시겠습니까? (Do you understand/know?)"
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "hv_come",
        "text": (
            "Use 오시다 instead of 오다 when the honored addressee is coming. "
            "Example: 오시겠습니까? (Will you come?)"
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "hv_see",
        "text": (
            "Use 보시다 instead of 보다, and 뵙다 (humble) when the speaker meets an honored "
            "person. Example: 뵙겠습니다 (I will see/meet you — humble)."
        ),
        "applies_to": ["formal"],
    },

    # ── Formal (합쇼체) sentence patterns ────────────────────────────────────

    {
        "id": "fp_endings",
        "text": (
            "Formal speech (합쇼체) requires -습니다/-ㅂ니다 for statements and -습니까/-ㅂ니까 "
            "for questions. Never mix these with -아요/-어요 endings in the same register."
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "fp_requests",
        "text": (
            "In formal speech, requests are softened with -해 주시겠습니까? (Would you please…) "
            "rather than direct imperatives."
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "fp_interview",
        "text": (
            "Job interviews and formal business presentations always require 합쇼체, "
            "even when the interviewer is younger than the candidate. Rank and setting "
            "override age in professional contexts."
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "fp_customer",
        "text": (
            "Customer service interactions (customer addressing staff, or staff addressing "
            "customer) use formal or polite speech. Casual speech is never appropriate "
            "in commercial service contexts."
        ),
        "applies_to": ["formal", "polite"],
    },
    {
        "id": "fp_you",
        "text": (
            "Avoid 당신 (당신 can sound confrontational) and 너 (너 is casual) in formal speech. "
            "Instead, address the person by title or role: 선생님, 사장님, 교수님, 부장님."
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "fp_written",
        "text": (
            "Formal written communication (official emails, reports, announcements) uses "
            "합쇼체. Polite or casual endings are never appropriate in formal documents."
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "fp_university",
        "text": (
            "University students addressing professors always use 합쇼체 regardless of "
            "the professor's apparent age or informal setting."
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "fp_academic_pres",
        "text": (
            "Academic presentations (thesis defense, conference talks) require 합쇼체 even "
            "when presenting to peers. The setting elevates the register."
        ),
        "applies_to": ["formal"],
    },

    # ── Polite (해요체) patterns ──────────────────────────────────────────────

    {
        "id": "pp_endings",
        "text": (
            "Polite speech (해요체) uses -아요/-어요/-여요 endings. These are respectful but "
            "less stiff than 합쇼체 — appropriate for most everyday interactions."
        ),
        "applies_to": ["polite"],
    },
    {
        "id": "pp_humble_i",
        "text": (
            "In polite speech when addressing elders or superiors, use 저 (humble 'I') "
            "rather than 나. Example: 저는 갈게요. (I will go.)"
        ),
        "applies_to": ["polite", "formal"],
    },
    {
        "id": "pp_requests",
        "text": (
            "Polite requests use -해 주세요 (please do X) or -해 줄 수 있어요? (can you do X?). "
            "These are softer than formal -해 주시겠습니까?"
        ),
        "applies_to": ["polite"],
    },
    {
        "id": "pp_colleagues",
        "text": (
            "Co-workers of similar rank typically use 해요체 unless they have explicitly "
            "agreed to use banmal. Default to polite with new colleagues."
        ),
        "applies_to": ["polite"],
    },
    {
        "id": "pp_medical",
        "text": (
            "Patients addressing doctors use 합쇼체; doctors to adult patients typically "
            "use 해요체. 반말 from a doctor is only appropriate with children."
        ),
        "applies_to": ["polite", "formal"],
    },

    # ── Casual (해체 / 반말) patterns ─────────────────────────────────────────

    {
        "id": "cp_endings",
        "text": (
            "Casual speech (해체 / 반말) drops the -요 suffix. 먹어요 → 먹어; 갈게요 → 갈게; "
            "봐요 → 봐. Sentence-final endings: -아, -어, -냐, -지, -구나."
        ),
        "applies_to": ["casual"],
    },
    {
        "id": "cp_questions",
        "text": (
            "Casual questions use rising intonation with -어?/-아?: 밥 먹었어? (Did you eat?) "
            "어디 가? (Where are you going?) 알아? (Do you know?)"
        ),
        "applies_to": ["casual"],
    },
    {
        "id": "cp_pronoun",
        "text": (
            "In casual speech, use 나 for 'I' (not the humble 저). "
            "Example: 나 지금 바빠 (I'm busy right now)."
        ),
        "applies_to": ["casual"],
    },
    {
        "id": "cp_invitation",
        "text": (
            "Casual invitations use the -자 form: 같이 밥 먹자 (Let's eat together). "
            "This is equivalent to '-(으)ㄹ까요?' but without any formality."
        ),
        "applies_to": ["casual"],
    },
    {
        "id": "cp_close_friends",
        "text": (
            "Between close friends of similar age, 해체 (반말) is expected and natural. "
            "Using 해요체 with a close friend can feel overly formal or create distance."
        ),
        "applies_to": ["casual"],
    },

    # ── Age, seniority, and relationship nuance ───────────────────────────────

    {
        "id": "age_elder",
        "text": (
            "When addressing someone 10+ years older even in a casual social setting, "
            "해요체 is the minimum baseline. Pure 반말 to an elder is rude unless the "
            "elder explicitly invites it."
        ),
        "applies_to": ["polite", "formal"],
    },
    {
        "id": "age_sunbae",
        "text": (
            "Seniors (선배) in Korean social or educational contexts receive at least "
            "해요체. Even in relaxed social settings, 선배 outrank 후배 in register."
        ),
        "applies_to": ["polite"],
    },
    {
        "id": "age_ambiguous",
        "text": (
            "When relationship status or age is ambiguous (first meeting, new acquaintance), "
            "default to 해요체. Switching to 반말 should be mutually agreed upon."
        ),
        "applies_to": ["polite"],
    },
    {
        "id": "age_siblings",
        "text": (
            "Younger siblings use 해요체 or 합쇼체 when addressing older siblings. "
            "Siblings of similar age may use 반말 with each other."
        ),
        "applies_to": ["polite", "formal"],
    },

    # ── Morphological notes ───────────────────────────────────────────────────

    {
        "id": "morph_honorific_infix",
        "text": (
            "Korean verbs honor the subject via the -(으)시- infix before the final ending: "
            "가다 → 가시다 (formal/polite, subject honored); 먹다 → 드시다 (honorific stem)."
        ),
        "applies_to": ["formal", "polite"],
    },
    {
        "id": "morph_topic_particle",
        "text": (
            "The topic particle 은/는 is neutral; subject particles 이/가 are more neutral. "
            "In formal speech, subject marking is often omitted when clear from context."
        ),
        "applies_to": ["formal"],
    },
    {
        "id": "morph_question_form",
        "text": (
            "Formal yes/no questions end in -습니까/-ㅂ니까. "
            "Polite yes/no questions end in -아요?/-어요? with rising intonation. "
            "Casual yes/no questions end in -아?/-어? or -냐?"
        ),
        "applies_to": ["formal", "polite", "casual"],
    },
]

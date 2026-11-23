"""Verification text: every label, note, verdict string and heading of the integrated verification, in two languages

PURPOSE: One table of the strings of ``run_full_verification`` (summary checks, notes, report verdicts, printed
    headings) in English and in the old TF project's Arabic, and the ``Variant`` options that make the numbers match it.
TAGS: verification text, languages, old project parity, variant options, student t width, timestamp blocks
PITFALLS: ``lang="ar"`` is the old project's text character for character (the parity test compares the printed text
    and the summary table with the old project's); a new string needs BOTH languages; the verdict symbols are the
    same in both and the logic reads the symbol, never the words; ``OLD_VARIANT`` + ``OLD_COST_PER_SIDE`` are the whole
    list of differences the old project had (cost, stop width, bootstrap unit, candle draws, label clip, language).
SOURCE: strings of crypto-signal-prediction workflow/verification.py, selective_eval.py and candle_baseline.py.
"""

from dataclasses import dataclass

__all__ = ["Variant", "DEFAULT_VARIANT", "OLD_VARIANT", "OLD_COST_PER_SIDE", "LANGS", "t"]

LANGS = ("en", "ar")
#: the old project's trade cost: 0.08 % for the ROUND TRIP = 0.04 % per side (``core.costs`` is per side, 0.08 %)
OLD_COST_PER_SIDE = 0.0004


@dataclass(frozen=True)
class Variant:
    """The choices the old project made differently; the defaults are this project's. ``stop_width``: the distance of
    the "nig" stop is the NIG predictive sd ("predictive_std") or the Student-t scale ("student_t", the old one);
    ``block``: the unit of every bootstrap and t-stat, the UTC calendar "day" or the raw "timestamp" (the old one);
    ``candle_n_boot``: draws of the candle report (None = the verification's ``n_boot``; the old one used 300);
    ``label_clip``: None = rows at the clip are left out of the label check, a number = compare every row against a
    return clipped to +-that (the old one: 1.0); ``lang``: "en" or "ar" (the old text)."""
    lang: str = "en"
    stop_width: str = "predictive_std"
    block: str = "day"
    candle_n_boot: int | None = None
    label_clip: float | None = None

    def __post_init__(self):
        if self.lang not in LANGS:
            raise ValueError(f"Variant.lang must be one of {LANGS}, got {self.lang!r}")
        if self.stop_width not in ("predictive_std", "student_t"):
            raise ValueError(f"Variant.stop_width must be 'predictive_std' or 'student_t', got {self.stop_width!r}")
        if self.block not in ("day", "timestamp"):
            raise ValueError(f"Variant.block must be 'day' or 'timestamp', got {self.block!r}")


DEFAULT_VARIANT = Variant()
OLD_VARIANT = Variant(lang="ar", stop_width="student_t", block="timestamp", candle_n_boot=300, label_clip=1.0)

_UP_NOTE_EN = "majority baseline = max(p, 1 - p)"
TEXT = {
    "en": {
        "sec_a": "a) data", "sec_b": "b) baselines", "sec_c": "c) confidence and trading", "sec_conf": "c) confidence",
        "sec_trade": "c) trading", "sec_d": "d) candle shape",
        "finite": "X_{stage} has no NaN / Inf", "labels": "labels match the raw prices ({stage})",
        "labels_skip": "skipped: target mode {mode!r} labels are not plain returns",
        "gap": "{a} before {b} in time (gap in bars)", "gap_note": "needs >= window + horizon = {n}",
        "up_share": "up share in train ({t})", "up_note": _UP_NOTE_EN, "days": "independent test days",
        "days_note": "every verdict below is on these days: fewer than ~250 means statistically weak verdicts",
        "dir": "direction accuracy {t} - majority class (test)",
        "dir_note": "model {acc:.4f} | majority {maj:.4f} | interval [{lo:+.4f}, {hi:+.4f}]",
        "mae": "MAE constant - MAE model ({t}_reg, test)",
        "mae_note": "model MAE {m:.5f} | constant ({c:+.5f}) {mc:.5f} | interval [{lo:+.5f}, {hi:+.5f}]",
        "skipped": "skipped", "c_skip_mode": "assumes 'return' targets: target mode {mode!r}",
        "c_skip_close": "no close head in the run's tables",
        "c_skip_trade": "{stop!r} trades need the high and low heads in the run",
        "d_skip": "no high / low class head in the run's tables",
        "consistency": "confidence consistency ({score})", "consistency_note": "needs > +0.5 on both periods",
        "slice": "slice >= {acc:.0%} ({score})", "trades": "trades {rr:g}:1 ({score})",
        "candle": "{t}_class above {best}",
        "sel_pass": "held on test", "sel_fail": "did not hold on test", "sel_none": "no val slice reached {acc:.0%}",
        "tr_pass": "earned on val and test, significant over days, beats a random direction",
        "tr_fail": "did not hold (needs val_exp_R > 0, test_exp_R > 0, test_t_days >= 2)",
        "tr_few": "fewer than {n} trades on val",
        "cv_pass": "the model adds information over {best} (combined dAUC {di:+.4f}, interval [{lo:+.4f}, {hi:+.4f}])",
        "cv_fail": ("{best} equals or beats the model: it adds nothing over the candle shape (combined dAUC "
                    "{di:+.4f} < {gain})"),
        "cv_flat": "no significant difference from {best}: no evidence of added value",
        "p_sel_title": ("\n\U0001f3af selective direction accuracy (target >= {acc:.0%} on test, "
                        "threshold from val only)"),
        "p_sel_1": ("   monotonic_* = Spearman between the confidence slice and its accuracy: near +1 consistent, "
                    "near 0 confidence means nothing"),
        "p_sel_2": ("   ⚠️ the Wilson interval assumes independent rows; coins of one day are not: read "
                    "test_dates (the real number of days), not test_n alone"),
        "p_tr_title": ("\n\U0001f4b9 {rr:g}:1 trades (stop={stop}, cost {cost}% round trip, threshold from val only)"),
        "p_tr_1": ("   win = share that really reached the target; exp_R = mean profit in units of risk after cost "
                   "(> 0 = profitable); both_hit = touched target and stop together, counted a loss"),
        "p_cb_title": ("\n\U0001f56f️ high / low heads against baselines from the candle shape (every baseline "
                       "trained on train only)"),
        "p_cb_check": "   label definition from the raw prices - val: {val} | test: {test}",
        "p_cb_maj": "\n   majority_baseline: {d}",
        "p_cb_best": ("\n   [{t}] best baseline (chosen on val): {best} | model AUC - baseline on test: {d:+.4f} "
                      "[{lo:+.4f}, {hi:+.4f}] | test days: {days}"),
        "p_banner": "INTEGRATED VERIFICATION", "p_counts": "\n   ✅ {ok} | ❌ {bad} | ➖/⚠️/ℹ️ {rest}",
        "p_end_ok": ("at least one confidence / trade source passed test: check it on other time windows before any "
                     "use"),
        "p_end_bad": "nothing tradeable passed test with this model and this data", "p_concl": "   conclusion:",
        "p_saved": "   \U0001f4be saved to {out}/verification_summary.csv and .json",
    },
    "ar": {
        "sec_a": "أ) البيانات", "sec_b": "ب) خطوط الأساس", "sec_c": "ج) الثقة والتداول", "sec_conf": "ج) الثقة",
        "sec_trade": "ج) التداول", "sec_d": "د) شكل الشمعة",
        "finite": "X_{stage} بلا NaN/Inf", "labels": "الأهداف تطابق الأسعار الخام ({stage})",
        "labels_skip": "تخطٍّ: TARGET_MODE={mode!r} (فحصه الذاتي في القسم ٣-ب)",
        "gap": "{a} قبل {b} زمنياً (فجوة بالشموع)", "gap_note": "المطلوب ≥ نافذة+أفق = {n}",
        "up_share": "نسبة الصعود في train ({t})", "up_note": "خط أساس الفئة الأكبر = max(p, 1−p)",
        "days": "أيام test المستقلة",
        "days_note": "كل الأحكام أدناه على هذه الأيام — أقل من ~250 يوماً يعني أحكاماً ضعيفة إحصائياً",
        "dir": "دقة اتجاه {t} − الفئة الأكبر (test)",
        "dir_note": "دقة النموذج {acc:.4f} | الفئة الأكبر {maj:.4f} | فاصل [{lo:+.4f}, {hi:+.4f}]",
        "mae": "MAE الثابت − MAE النموذج ({t}_reg، test)",
        "mae_note": "MAE النموذج {m:.5f} | الثابت ({c:+.5f}) {mc:.5f} | فاصل [{lo:+.5f}, {hi:+.5f}]",
        "skipped": "تخطٍّ", "c_skip_mode": "يفترض أهداف 'return' — TARGET_MODE={mode!r}",
        "c_skip_close": "لا رأس close في جداول التشغيل",
        "c_skip_trade": "صفقات {stop!r} تحتاج رأسَي high وlow في التشغيل",
        "d_skip": "لا رأس تصنيف high / low في جداول التشغيل",
        "consistency": "اتساق الثقة ({score})", "consistency_note": "المطلوب > +0.5 على الفترتين معاً",
        "slice": "شريحة ≥ {acc:.0%} ({score})", "trades": "صفقات {rr:g}:1 ({score})",
        "candle": "{t}_class فوق {best}",
        "sel_pass": "صمد على test", "sel_fail": "لم يصمد على test", "sel_none": "لا شريحة على val بلغت {acc:.0%}",
        "tr_pass": "ربح على val وtest، دالّ على مستوى الأيام، وأفضل من اتجاه عشوائي",
        "tr_fail": "لم يصمد (يلزم: val_exp_R>0 و test_exp_R>0 و test_t_days≥2)",
        "tr_few": "أقل من {n} صفقة على val",
        "cv_pass": "النموذج يضيف معلومة فوق {best} (ΔAUC عند الدمج {di:+.4f}، فاصل [{lo:+.4f}, {hi:+.4f}])",
        "cv_fail": ("{best} يعادل النموذج أو يتفوّق عليه — النموذج لا يضيف شيئاً فوق شكل الشمعة "
                    "(ΔAUC عند الدمج {di:+.4f} < {gain})"),
        "cv_flat": "لا فرق دالّ عن {best} — لا دليل على قيمة مضافة",
        "p_sel_title": "\n\U0001f3af دقة الاتجاه الانتقائية (الهدف ≥ {acc:.0%} على test، العتبة من val فقط)",
        "p_sel_1": "   monotonic_* = سبيرمان بين شريحة الثقة ودقتها: قرب +1 اتساق، قرب 0 الثقة لا تعني شيئاً",
        "p_sel_2": ("   ⚠️ فاصل Wilson يفترض عيّنات مستقلة؛ العملات المترابطة في نفس اليوم ليست كذلك —"
                    " انظر test_dates (عدد الأيام الفعلية) لا test_n وحده"),
        "p_tr_title": "\n\U0001f4b9 صفقات {rr:g}:1 (وقف={stop}, تكلفة {cost}% ذهاباً وإياباً، العتبة من val فقط)",
        "p_tr_1": ("   win = نسبة بلوغ الهدف فعلاً؛ exp_R = متوسط الربح بوحدات المخاطرة بعد التكلفة (> 0 = مربح)؛ "
                   "both_hit = لمس الهدف والوقف معاً، حُسب خسارة"),
        "p_cb_title": ("\n\U0001f56f️ رأسا high/low مقابل خطوط أساس من شكل الشمعة (كل خطوط الأساس دُرِّبت على "
                       "train فقط)"),
        "p_cb_check": "   فحص تعريف الهدف من الأسعار الخام — val: {val} | test: {test}",
        "p_cb_maj": "\n   majority_baseline: {d}",
        "p_cb_best": ("\n   [{t}] أفضل خط أساس (مختار على val): {best} | AUC النموذج − الأساس على test: {d:+.4f} "
                      "[{lo:+.4f}, {hi:+.4f}] | أيام test: {days}"),
        "p_banner": "\U0001f9fe ملخص التحقق المتكامل",
        "p_counts": "\n   ✅ {ok} | ❌ {bad} | ➖/⚠️/ℹ️ {rest}",
        "p_end_ok": "يوجد على الأقل مصدر ثقة/صفقات اجتاز test — افحصه على نوافذ زمنية أخرى قبل أي استخدام",
        "p_end_bad": "لا شيء قابل للتداول اجتاز test بهذا النموذج وهذه البيانات", "p_concl": "   الخلاصة:",
        "p_saved": "   \U0001f4be حُفظ في {out}/verification_summary.csv و.json",
    },
}


def t(lang: str, key: str, **kw) -> str:
    """The string ``key`` of ``lang`` formatted with ``kw``; an unknown language or key raises."""
    if lang not in TEXT:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    return TEXT[lang][key].format(**kw)

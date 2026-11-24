"""PURPOSE: Every printed sentence of the chicks report (headings, verdicts, warnings, table notes) in the old TF
    project's Arabic and in English, and ``Say``, the small printer that formats one by key and language.
TAGS: chicks text, languages, arabic, english, say, printed report, tf parity, message catalog
PITFALLS: ``lang="ar"`` is the old project's text character for character (the parity test compares the printed
    output with the old code's); a new message needs BOTH languages (``tests`` check the two key sets are equal); a
    key that is not in the catalog raises, never prints a blank; the data words inside the result tables (the Arabic
    column names such as ``عدد`` / ``نسبة_نجاح`` and the direction words) are NOT here: they are table identifiers
    that the code and the figures read by name and stay Arabic whatever ``lang`` is; figure titles are English in
    both languages (matplotlib cannot shape Arabic without ``arabic_reshaper``).
SOURCE: the strings of crypto-signal-prediction ``evaluation/`` (all_assets, unified, latest_table, trust_calibration,
    output_conditioned, trade_selection, patterns, integrity, tearsheet, outputs, full_analysis).
"""

__all__ = ["LANGS", "TEXT", "Say", "t"]

LANGS = ("ar", "en")



def _sig_head(strategy: str, interval: str, half1: str, half2: str) -> str:
    """The header of the significance table, padded to the widths of its rows."""
    return f"{strategy:<12} {'Sharpe':>7} {interval:>17} {'t':>6} {'p (t)':>7} {half1:>9} {half2:>9}"


_SIG_HEAD_AR = _sig_head("الاستراتيجية", "فترة ثقة 95%", "نصف1 Σ%", "نصف2 Σ%")
_SIG_HEAD_EN = _sig_head("strategy", "95% interval", "half1 Σ%", "half2 Σ%")

AR = {
    # output manager
    "out_df_big": "📄 النتيجة كبيرة ({n:,} صف) → حُفظت كاملة في: {path}",
    "out_df_preview": "   معاينة أول {k} صف:",
    "out_df_more": "   ... ({n:,} صف إضافي في الملف)",
    "out_txt_big": "📄 النص طويل ({n:,} حرف) → حُفظ كاملاً في: {path}",
    "out_txt_cut": "   ... [تم الاقتصاص هنا — راجع الملف للنص الكامل]",
    # per coin
    "tested_head": "🧪 اختبار {n} عملة (v4.2) — يُعرض آخر {k} عينة لكل هدف",
    "targets_line": "🎯 الأهداف المُقيَّمة: {names}",
    "asset_head": "[{i}/{n}] 🪙 {asset}  ({rows:,} عينة)",
    "latest_head": "\n📊 آخر {k} عينة لكل هدف (الأحدث في الأسفل):",
    "perf_head": "\n📈 الأداء على كامل العينات:",
    "perf_line": "   ▸ {name:<6}│ MAE {mae} │ MAPE {mape:.4f}% │ Win {win:.2f}% │ corr {corr:+.3f} │ n={n:,}",
    "asset_error": "   ❌ خطأ في {asset}: {e}",
    "none_tested": "\n⚠️ لم يتم اختبار أي عملة بنجاح",
    "latest_legend": ("signal: ▲ صعود متوقع | ▼ هبوط  ·  conf: الثقة  ·  unc: عدم اليقين كنسبة من السعر المتوقع"
                      "  ·  move: التحرك المتوقع عن سعر الدخول"),
    "latest_legend_hourglass": "  ·  ⏳: لا يوجد هدف فعلي بعد",
    "latest_none": "(لا توجد عينات للعرض)",
    # verification, coverage, totals
    "ver_head": "🔍 التحقق من فك التشفير — مرة واحدة لكل الأصول (round-trip / اتساق الفضاءين المطبّع والأصلي)",
    "ver_all_ok": "✅ جميع الأهداف والأصول اجتازت التحقق",
    "ver_failed": "❌ {n} حالة فشل — راجع verification_summary",
    "ver_note": ("   (mae_scaled / mae_real: متوسط مرجَّح بعدد العينات؛ mae_real يُقارَن فقط إن كانت أسعار الأصول "
                 "بمقياس متقارب — تفصيله لكل أصل في جدول الملخص أدناه)"),
    "status_ok": "✅ ok",
    "status_failed": "❌ {n} فشل",
    "cov_line": ("📐 التغطية: {ev:,} من {tr:,} صف اختبار | {ea} من {ta} عملة | صفوف/عملة: "
                 "min={lo}، وسيط={med:.0f}، max={hi}"),
    "cov_not_all": " (eval_all_rows=False: تُتخطّى العملات < {n} صف)",
    "cov_skipped": "   ⚠️ تُخطّيت {n} عملة بصفوف قليلة: {items}{more}",
    "cov_small_median": ("   ℹ️ وسيط صفوف العملة صغير: حجم قسم الاختبار نفسه (train_pct/val_pct + فجوة العزل) — "
                         "لا سقف في chicks؛ الدقة لكل عملة ضجيج (انظر فاصل Wilson) والمعتمد هو المجمَّع."),
    "asset_summary_head": "📋 ملخص الأداء لكل أصل (MAE بوحدة سعر الأصل، win% = صحة الاتجاه)",
    "asset_ci_head": "📋 دقة كل عملة × هدف مع n وفاصل Wilson 95% (الجدول كاملاً في analysis_outputs/ إن كبر)",
    "overall_head": "🧮 الإجمالي لكل هدف عبر كل الأصول (مجمّع بالعينات)",
    # trust and calibration
    "trust_title": "🧠 تقرير الثقة والفهم (Trust & Calibration Report)",
    "trust_overall": "\n📊 التقييم العام (كل البيانات):",
    "trust_verdict_head": "\n💡 الحكم العام:",
    "trust_good": "   ✅ النموذج 'يفهم' توقعاته جيداً (Trust Score: {s:.1f}/100) — يمكن الاعتماد على الثقة كمرشّح للجودة",
    "trust_mid": "   ⚠️  فهم متوسط (Trust Score: {s:.1f}/100) — الثقة مفيدة جزئياً لكنها تحتاج معايرة",
    "trust_bad": "   ❌ فهم ضعيف (Trust Score: {s:.1f}/100) — لا يُنصح بالاعتماد على درجة الثقة كما هي",
    "ece_good": "      • المعايرة (ECE={e:.3f}) ممتازة",
    "ece_mid": "      • المعايرة (ECE={e:.3f}) مقبولة",
    "ece_bad": "      • المعايرة (ECE={e:.3f}) ضعيفة — النموذج مبالغ/متحفّظ في ثقته",
    "trust_by_group": "\n📋 التفصيل حسب '{g}':",
    "calib_fit": "🎚️ معايرة [{g}] ({method}) على val n={n}: الإشارة = {signal}",
    "calib_warn": "  ⚠️ رأس الثقة {why} على val (std={std:.4f}، قيم فريدة={uniq}، rho={rho:.3f}) ← {to}",
    "calib_const": "ثابت فعلياً",
    "calib_nosig": "بلا ارتباط موجب دالّ بالصحة",
    "calib_to_margin": "المعايرة من |p_up−0.5| لرأس التصنيف",
    "calib_to_base": "نسبة الصحة الثابتة فقط",
    "calib_head": "\n🎚️ معايرة الثقة post-hoc (مُلائَمة على val، مُطبَّقة على test) — ECE قبل ← بعد",
    "calib_margin_note": ("   ⚠️ [{t}] رأس الثقة بلا معلومة ← المعايرة من |p_up−0.5| لرأس التصنيف؛ ECE هنا لصحة "
                          "اتجاه رأس التصنيف (لا رأس الانحدار). "),
    "calib_base_note": "   ⚠️ [{t}] لا إشارة ثقة مفيدة (ولا p_up): المعايرة = نسبة الصحة الثابتة على val.",
    "calib_footer": ("   ℹ️ المعايرة تُصلح الانحياز لا التمييز: هبوط ECE لا يعني أن الثقة صارت تفرّق الصحيح من "
                     "الخاطئ (راجع confidence_accuracy_corr في التقرير أعلاه)."),
    # output conditioned
    "move_no_col": "العمود '{c}' غير موجود — هذا التحليل يتطلب أهدافاً مستمرة (أسعار).",
    "move_title": "🎯 نسبة النجاح حسب نوع/حجم الحركة المتوقعة",
    "move_best": "\n✅ أفضل نطاق حركة: {t} / {d} {b} → نجاح {w}% ({n} عينة)",
    "move_worst": "❌ أسوأ نطاق حركة: {t} / {d} {b} → نجاح {w}% ({n} عينة)",
    "conf_title": "📊 نسبة النجاح حسب فئة الثقة",
    "batch_title": "📦 تحليل الأداء عبر الدفعات (batch_size={b}{t})",
    "batch_regression": "⚠️ انحدار",
    "batch_improved": "✅ تحسّن",
    "batch_normal": "طبيعي",
    "batch_anoms": "\n🔎 دفعات شاذة مقارنة بأفضل دفعة (#{b}, {w}% نجاح):",
    "batch_anom_line": "   • الدفعة #{b}: {w}% نجاح ({s}) — أقل من الأفضل بـ {d:.2f} نقطة مئوية",
    "batch_stable": "\n✅ لا توجد دفعات شاذة — الأداء مستقر عبر كل الدفعات",
    # trade selection
    "trade_none": "⚠️ لا توجد صفقات يتجاوز فيها |التحرك المتوقع| عدم اليقين × {k} (0 من {n}).",
    "trade_empty": "(لا توجد صفقات تجتاز الفلتر)",
    "trade_more": "… و{n} صفقة أخرى (max_rows=None لعرض الكل)",
    "trade_rates": ("📈 معدل الإصابة: قبل الفلترة {b:.1f}% (n={bn})  ->  بعد الفلترة + Top-N {s:.1f}% "
                    "(n={sn}, {share:.0f}% من الصفقات)"),
    # patterns
    "pat_too_few": "عدد العينات الصالحة قليل جداً (<20) لاكتشاف أنماط موثوقة.",
    "pat_head_tag": " — الرأس: {h}",
    "pat_title": "🔬 اكتشاف الأنماط آلياً في مخرجات النموذج{h}",
    "pat_tree": ("\n🌳 1) قواعد شجرة القرار (دقة خام: {a:.1f}%، خط أساس ساذج: {n:.1f}%، "
                 "دقة موزونة: {b:.1f}%):"),
    "pat_importance": "\n📊 2) أهمية الخصائص في التمييز بين النجاح والفشل:",
    "pat_top_feature": "   → أهم خاصية تفسّر نجاح/فشل التوقع: '{f}'",
    "pat_clusters": "\n🧩 3) الأنماط السلوكية المكتشفة (K-Means, k={k}):",
    "pat_best": "   ✅ أفضل نمط: العنقود #{c} بنجاح {w}%",
    "pat_worst": "   ❌ أسوأ نمط: العنقود #{c} بنجاح {w}%",
    "pat_cluster_fail": "   ⚠️ تعذر رسم العناقيد: {e}",
    "pat_anom_head": "\n🚨 4) التوقعات الشاذة (Isolation Forest): {a} من {n} ({p:.1f}%)",
    "pat_normal": "   • دقة التوقعات العادية: {a:.1f}%",
    "pat_anom": "   • دقة التوقعات الشاذة:  {a:.1f}%",
    "pat_anom_worse": "   ⚠️  التوقعات الشاذة أقل دقة بوضوح — قد تحتاج فلترة أو مراجعة يدوية",
    "pat_anom_better": "   💡 التوقعات الشاذة أعلى دقة رغم ندرتها — تستحق دراسة (فرصة محتملة)",
    "pat_anom_same": "   ℹ️  لا فرق واضح في الدقة بين الشاذ والعادي",
    "pat_head_skipped": "\n⚠️ الرأس {h}: تخطي اكتشاف الأنماط — {e}",
    "pat_none_left": "لا رأس فيه عيّنات صالحة كافية لاكتشاف الأنماط: {s}",
    "pat_summary_title": "🔬 ملخص اكتشاف الأنماط لكل رأس — خط الأساس الساذج (الفئة الأكثر) مقابل الدقة الموزونة",
    "pat_summary_note": ("   الدقة الموزونة تُقاس داخل العيّنة (الشجرة تُدرَّب وتُقاس على البيانات نفسها) وخط أساسها "
                         "الساذج 0.5؛ رأس دقته الخام بمستوى خط الأساس ودقته الموزونة قرب 0.5 لا نمط فيه."),
    # integrity
    "integ_none": "⚠️ لا توجد أهداف مستمرة كافية لتشخيص النزاهة",
    "integ_title": "🩺 تشخيص نزاهة النموذج (Naive Baseline × Lag Scan × الانحياز الموجّه)",
    "integ_healthy": "\n✅ عدد تركيبات (عملة/هدف) اجتازت كل الفحوصات: {h} / {n}",
    "integ_flagged": "⚠️ تركيبات تحتاج مراجعة (قد يكون النموذج يعتمد على الترابط الذاتي بدل تنبؤ حقيقي):",
    "integ_plots": "\n📊 رسوم بيانية لأسوأ {w} وأفضل {b} تركيبة (الباقي محفوظ في الجدول أعلاه فقط):",
    "bias_up": "متفائل (يبالغ بالارتفاع)",
    "bias_down": "متشائم (يبالغ بالانخفاض)",
    "bias_flat": "متوازن",
    # tearsheet verdict
    "tear_no_data": "⚠️ لا توجد بيانات كافية لحساب مقاييس الأداء",
    "tear_sharpe_ann": "   • Sharpe Ratio (سنوي): {v:.3f}  ({j})",
    "tear_good": "✅ جيد",
    "tear_weak": "⚠️ ضعيف",
    "tear_neg": "❌ سلبي",
    "tear_sharpe_trade": "   • Sharpe (لكل صفقة، غير سنوي): {v:.4f}",
    "tear_sortino": "   • Sortino Ratio:      {v:.3f}",
    "tear_dd": "   • Max Drawdown:       {v:.2f}%  ({j})",
    "tear_dd_ok": "✅ محتمل",
    "tear_dd_high": "⚠️ مرتفع",
    "tear_pf": "   • Profit Factor:      {v:.3f}  ({j})",
    "tear_pf_strong": "✅ قوي",
    "tear_pf_ok": "⚠️ مقبول",
    "tear_pf_bad": "❌ خاسر",
    "tear_omega": "   • Omega Ratio:        {v:.3f}",
    "tear_sqn": "   • SQN (Van Tharp):    {v:.3f}  ({j})",
    "tear_sqn_top": "✅ ممتاز",
    "tear_sqn_ok": "⚠️ قابل للتداول",
    "tear_sqn_weak": "❌ ضعيف",
    "tear_total": "   • إجمالي العائد:      {v:+.2f}% على {n} {u}",
    "tear_unit_period": "فترة (محفظة)",
    "tear_unit_trade": "صفقة",
    # significance
    "sig_title": "🧪 هل الربح إشارة أم حظ؟ (عوائد الفترات بعد التكلفة{mn})",
    "sig_mn": " ومحايدة للسوق",
    "sig_table_head": _SIG_HEAD_AR,
    "sig_model": "النموذج",
    "sig_momentum": "الزخم",
    "sig_reversal": "الارتداد",
    "sig_perm": ("\n   • اختبار التبديل ({n} خلطة لاتجاهات النموذج داخل كل فترة): متوسط الفترة {o:+.4f}% مقابل "
                 "{nl:+.4f}% للخلط (95%: {p95:+.4f}%) → p = {p:.3f}"),
    "sig_agree": "   • اتفاق النموذج مع قاعدة الارتداد: {a:.1f}% (50% = مستقل عنها، ~100% = يكرّرها)",
    "sig_skill": "   • المهارة (أفضل من اتجاهات عشوائية): {v} (p={p:.3f})",
    "sig_yes": "✅ نعم",
    "sig_no_skill": "❌ غير مُثبتة",
    "sig_profit": "   • الربح بعد التكلفة: {v} (t={t:.2f}؛ t≥2 يلزمه نحو {k} فترة بنفس الأداء)",
    "sig_sig": "✅ دالّ",
    "sig_not_sig": "⚠️ غير دالّ بعد — قد يكون حظاً؛ يلزم فترة أطول",
    "sig_rules": "   • مقابل القواعد البسيطة: {v}",
    "sig_beats": "✅ يتفوّق على الزخم والارتداد",
    "sig_not_beats": "⚠️ لا يتفوّق على كل قاعدة بسيطة",
    "sig_regime": "   • ⚠️ الربح كله في النصف الأول والنصف الثاني غير رابح — نظام انتهى أو حظ",
    "chicks_skipped": "⏭️ القسم ٦ متخطّى — أهداف test غير مدعومة في chicks (target_mode={mode!r})",
    # full analysis
    "full_no_results": "⚠️ لا توجد نتائج — تحقق من test_dict",
    "full_val_empty": "⚠️ val_dict بلا نتائج — تُتخطّى معايرة الثقة",
    "full_trade_title": "\n🎯 أفضل الصفقات — |التحرك المتوقع| > {k:.2f}× عدم اليقين، ترتيب: {s}",
    "full_trade_rate": "  📊 Win Rate بعد Edge Filter (كل المؤهلين): {a:.1f}% (n={n}) | قبل الفلترة: {b:.1f}% (n={bn})",
    "full_pat_skip": "\n⚠️ تخطي اكتشاف الأنماط: {e}",
    "full_basis_portfolio": "محفظة متساوية الأوزان: {p} فترة، {t} صفقة، بعد تكلفة {c}%",
    "full_basis_neutral": " — محايدة للسوق: عائد كل صفقة ناقص متوسط كل العملات في نفس الفترة",
    "full_basis_untimed": "⚠️ بلا طوابع زمنية: صفقات متتالية — ليس منحنى قابلاً للتنفيذ",
    "full_tear_title": "📈 مقاييس أداء تداول إضافية (Tearsheet-style) — هدف Close، كل العملات مجمّعة",
    "full_sig_fail": "⚠️ تعذّر تقرير الدلالة الإحصائية: {e}",
    "full_done": "✅ اكتمل التحليل الشامل — الملخص التنفيذي",
    "full_n": "   • إجمالي العينات المحللة: {n:,}",
    "full_acc": "   • الدقة/نسبة النجاح العامة: {a:.2f}%  (Wilson 95%: {lo:.1f}–{hi:.1f}%)",
    "full_cov": "   • التغطية: {ev:,} من {tr:,} صف اختبار، {ea} من {ta} عملة",
    "full_clipped": "   • عدم اليقين المقصوص بحدّ NIG_UNC_MAX={m}: {n:,} صف ({p:.2f}%) — عيّنات خارج التوزيع",
    "full_ece": "   • ECE قبل ← بعد المعايرة: {s}",
    "full_trades": "   • الصفقات المؤهلة بعد Edge Filter: {n:,} صفقة",
    "full_trust": "   • درجة الثقة والفهم (Trust Score): {s:.1f}/100",
    "full_decode_ok": "   • حالة التحقق من فك التشفير: ✅ سليم بالكامل",
    "full_decode_bad": "   • حالة التحقق من فك التشفير: ❌ {n} فشل",
    "full_beats_naive": "   • نسبة التركيبات (عملة/هدف) التي تتفوق فعلياً على Naive Baseline: {p:.0f}%",
    "full_sharpe": "   • Sharpe Ratio (Close، {b}): {s:.2f} | Max Drawdown: {d:.1f}%",
    "full_ann": "سنوي",
    "full_per_trade": "لكل صفقة",
    "full_saved": "\n📁 كل الجداول/الرسوم الكبيرة محفوظة في: {p}",
}

EN = {
    "out_df_big": "📄 The result is big ({n:,} rows) → saved whole to: {path}",
    "out_df_preview": "   preview of the first {k} rows:",
    "out_df_more": "   ... ({n:,} more rows in the file)",
    "out_txt_big": "📄 The text is long ({n:,} characters) → saved whole to: {path}",
    "out_txt_cut": "   ... [cut here — see the file for the full text]",
    "tested_head": "🧪 Testing {n} coins (v4.2) — the last {k} sample(s) of each target are shown",
    "targets_line": "🎯 Evaluated targets: {names}",
    "asset_head": "[{i}/{n}] 🪙 {asset}  ({rows:,} samples)",
    "latest_head": "\n📊 Last {k} sample(s) of each target (newest at the bottom):",
    "perf_head": "\n📈 Performance on all samples:",
    "perf_line": "   ▸ {name:<6}│ MAE {mae} │ MAPE {mape:.4f}% │ Win {win:.2f}% │ corr {corr:+.3f} │ n={n:,}",
    "asset_error": "   ❌ error in {asset}: {e}",
    "none_tested": "\n⚠️ No coin was tested successfully",
    "latest_legend": ("signal: ▲ up expected | ▼ down  ·  conf: confidence  ·  unc: uncertainty as a share of the "
                      "predicted price  ·  move: expected move from the entry price"),
    "latest_legend_hourglass": "  ·  ⏳: no true target yet",
    "latest_none": "(no samples to show)",
    "ver_head": "🔍 Decoding verification — once for all assets (round trip / normalised vs original space)",
    "ver_all_ok": "✅ every target and asset passed the verification",
    "ver_failed": "❌ {n} failures — see verification_summary",
    "ver_note": ("   (mae_scaled / mae_real: mean weighted by samples; mae_real is comparable only when the asset "
                 "prices are on a similar scale — per-asset detail in the summary table below)"),
    "status_ok": "✅ ok",
    "status_failed": "❌ {n} failed",
    "cov_line": ("📐 Coverage: {ev:,} of {tr:,} test rows | {ea} of {ta} coins | rows per coin: "
                 "min={lo}, median={med:.0f}, max={hi}"),
    "cov_not_all": " (eval_all_rows=False: coins with < {n} rows are skipped)",
    "cov_skipped": "   ⚠️ {n} coins with few rows were skipped: {items}{more}",
    "cov_small_median": ("   ℹ️ the median rows per coin is small: it is the size of the test split itself "
                         "(train_pct/val_pct + the purge gap), there is no cap in chicks; the accuracy of one coin is "
                         "noise (see the Wilson interval), the pooled number is the one to use."),
    "asset_summary_head": "📋 Performance per asset (MAE in the asset's price unit, win% = direction hit)",
    "asset_ci_head": "📋 Accuracy per coin x target with n and a Wilson 95% interval (the whole table is in the output "
                     "folder when long)",
    "overall_head": "🧮 Total per target over all assets (pooled by samples)",
    "trust_title": "🧠 Trust & Calibration Report",
    "trust_overall": "\n📊 Overall assessment (all data):",
    "trust_verdict_head": "\n💡 Overall verdict:",
    "trust_good": "   ✅ The model 'understands' its predictions well (Trust Score: {s:.1f}/100) — the confidence can "
                  "be used as a quality filter",
    "trust_mid": "   ⚠️  Medium understanding (Trust Score: {s:.1f}/100) — the confidence is partly useful but needs "
                 "calibration",
    "trust_bad": "   ❌ Weak understanding (Trust Score: {s:.1f}/100) — do not rely on the confidence as it is",
    "ece_good": "      • calibration (ECE={e:.3f}) excellent",
    "ece_mid": "      • calibration (ECE={e:.3f}) acceptable",
    "ece_bad": "      • calibration (ECE={e:.3f}) weak — the model is over- or under-confident",
    "trust_by_group": "\n📋 Breakdown by '{g}':",
    "calib_fit": "🎚️ calibration [{g}] ({method}) on val n={n}: signal = {signal}",
    "calib_warn": "  ⚠️ the confidence head {why} on val (std={std:.4f}, unique values={uniq}, rho={rho:.3f}) ← {to}",
    "calib_const": "is effectively constant",
    "calib_nosig": "has no significant positive correlation with being right",
    "calib_to_margin": "calibrated from |p_up−0.5| of the class head",
    "calib_to_base": "the constant val accuracy only",
    "calib_head": "\n🎚️ Post-hoc confidence calibration (fitted on val, applied to test) — ECE before ← after",
    "calib_margin_note": ("   ⚠️ [{t}] the confidence head carries no information ← calibrated from |p_up−0.5| of the "
                          "class head; the ECE here is for the class head's direction hit (not the regression "
                          "head's). "),
    "calib_base_note": "   ⚠️ [{t}] no useful confidence signal (and no p_up): the calibration = the constant val "
                       "accuracy.",
    "calib_footer": ("   ℹ️ Calibration repairs bias, not discrimination: a lower ECE does not mean the confidence now "
                     "separates right from wrong (see confidence_accuracy_corr in the report above)."),
    "move_no_col": "column '{c}' not found — this analysis needs continuous (price) targets.",
    "move_title": "🎯 Hit rate by the kind and size of the predicted move",
    "move_best": "\n✅ best move range: {t} / {d} {b} → hit {w}% ({n} samples)",
    "move_worst": "❌ worst move range: {t} / {d} {b} → hit {w}% ({n} samples)",
    "conf_title": "📊 Hit rate by confidence bucket",
    "batch_title": "📦 Performance across batches (batch_size={b}{t})",
    "batch_regression": "⚠️ regression",
    "batch_improved": "✅ improved",
    "batch_normal": "normal",
    "batch_anoms": "\n🔎 unusual batches against the best batch (#{b}, {w}% hit):",
    "batch_anom_line": "   • batch #{b}: {w}% hit ({s}) — {d:.2f} points below the best",
    "batch_stable": "\n✅ no unusual batches — performance is stable across all batches",
    "trade_none": "⚠️ no trades where |expected move| exceeds the uncertainty x {k} (0 of {n}).",
    "trade_empty": "(no trades pass the filter)",
    "trade_more": "… and {n} more trades (max_rows=None shows all)",
    "trade_rates": ("📈 hit rate: before the filter {b:.1f}% (n={bn})  ->  after the filter + Top-N {s:.1f}% "
                    "(n={sn}, {share:.0f}% of the trades)"),
    "pat_too_few": "too few valid samples (<20) to find reliable patterns.",
    "pat_head_tag": " — head: {h}",
    "pat_title": "🔬 Automatic pattern discovery in the model outputs{h}",
    "pat_tree": ("\n🌳 1) Decision-tree rules (raw accuracy: {a:.1f}%, naive baseline: {n:.1f}%, balanced "
                 "accuracy: {b:.1f}%):"),
    "pat_importance": "\n📊 2) Feature importance in telling success from failure:",
    "pat_top_feature": "   → the feature that best explains success / failure of a prediction: '{f}'",
    "pat_clusters": "\n🧩 3) Behaviour patterns found (K-Means, k={k}):",
    "pat_best": "   ✅ best pattern: cluster #{c} with {w}% hit",
    "pat_worst": "   ❌ worst pattern: cluster #{c} with {w}% hit",
    "pat_cluster_fail": "   ⚠️ could not draw the clusters: {e}",
    "pat_anom_head": "\n🚨 4) Anomalous predictions (Isolation Forest): {a} of {n} ({p:.1f}%)",
    "pat_normal": "   • accuracy of the normal predictions: {a:.1f}%",
    "pat_anom": "   • accuracy of the anomalous predictions:  {a:.1f}%",
    "pat_anom_worse": "   ⚠️  the anomalous predictions are clearly less accurate — they may need filtering or a manual "
                      "look",
    "pat_anom_better": "   💡 the anomalous predictions are more accurate despite being rare — worth studying",
    "pat_anom_same": "   ℹ️  no clear accuracy difference between anomalous and normal",
    "pat_head_skipped": "\n⚠️ head {h}: pattern discovery skipped — {e}",
    "pat_none_left": "no head has enough valid samples for pattern discovery: {s}",
    "pat_summary_title": "🔬 Pattern discovery summary per head — naive baseline (majority class) vs balanced accuracy",
    "pat_summary_note": ("   The balanced accuracy is measured in-sample (the tree is trained and scored on the same "
                         "data) and its naive baseline is 0.5; a head whose raw accuracy is at the baseline and whose "
                         "balanced accuracy is near 0.5 has no pattern."),
    "integ_none": "⚠️ no continuous targets with enough samples for the integrity diagnosis",
    "integ_title": "🩺 Model integrity diagnosis (Naive Baseline × Lag Scan × directional bias)",
    "integ_healthy": "\n✅ coin / target combinations that passed every check: {h} / {n}",
    "integ_flagged": "⚠️ combinations to review (the model may lean on autocorrelation instead of real prediction):",
    "integ_plots": "\n📊 figures for the worst {w} and best {b} combination(s) (the rest is only in the table above):",
    "bias_up": "optimistic (overshoots upward)",
    "bias_down": "pessimistic (overshoots downward)",
    "bias_flat": "balanced",
    "tear_no_data": "⚠️ not enough data to compute the performance metrics",
    "tear_sharpe_ann": "   • Sharpe Ratio (annualised): {v:.3f}  ({j})",
    "tear_good": "✅ good",
    "tear_weak": "⚠️ weak",
    "tear_neg": "❌ negative",
    "tear_sharpe_trade": "   • Sharpe (per trade, not annualised): {v:.4f}",
    "tear_sortino": "   • Sortino Ratio:      {v:.3f}",
    "tear_dd": "   • Max Drawdown:       {v:.2f}%  ({j})",
    "tear_dd_ok": "✅ tolerable",
    "tear_dd_high": "⚠️ high",
    "tear_pf": "   • Profit Factor:      {v:.3f}  ({j})",
    "tear_pf_strong": "✅ strong",
    "tear_pf_ok": "⚠️ acceptable",
    "tear_pf_bad": "❌ losing",
    "tear_omega": "   • Omega Ratio:        {v:.3f}",
    "tear_sqn": "   • SQN (Van Tharp):    {v:.3f}  ({j})",
    "tear_sqn_top": "✅ excellent",
    "tear_sqn_ok": "⚠️ tradable",
    "tear_sqn_weak": "❌ weak",
    "tear_total": "   • total return:       {v:+.2f}% over {n} {u}",
    "tear_unit_period": "periods (portfolio)",
    "tear_unit_trade": "trades",
    "sig_title": "🧪 Is the profit a signal or luck? (period returns after cost{mn})",
    "sig_mn": " and market-neutral",
    "sig_table_head": _SIG_HEAD_EN,
    "sig_model": "model",
    "sig_momentum": "momentum",
    "sig_reversal": "reversal",
    "sig_perm": ("\n   • permutation test ({n} shuffles of the model's directions inside each period): mean period "
                 "{o:+.4f}% against {nl:+.4f}% for the shuffles (95%: {p95:+.4f}%) → p = {p:.3f}"),
    "sig_agree": "   • agreement of the model with the reversal rule: {a:.1f}% (50% = independent, ~100% = repeats it)",
    "sig_skill": "   • skill (better than random directions): {v} (p={p:.3f})",
    "sig_yes": "✅ yes",
    "sig_no_skill": "❌ not shown",
    "sig_profit": "   • profit after cost: {v} (t={t:.2f}; t>=2 would take about {k} periods at the same performance)",
    "sig_sig": "✅ significant",
    "sig_not_sig": "⚠️ not significant yet — may be luck; a longer period is needed",
    "sig_rules": "   • against the simple rules: {v}",
    "sig_beats": "✅ beats momentum and reversal",
    "sig_not_beats": "⚠️ does not beat every simple rule",
    "sig_regime": "   • ⚠️ all the profit is in the first half and the second half does not earn — a regime that ended, "
                  "or luck",
    "chicks_skipped": "⏭️ chicks skipped — the target mode {mode!r} does not decode to a price forecast",
    "full_no_results": "⚠️ no results — check the stage",
    "full_val_empty": "⚠️ the validation stage has no results — the confidence calibration is skipped",
    "full_trade_title": "\n🎯 Best trades — |expected move| > {k:.2f}x uncertainty, ranked by: {s}",
    "full_trade_rate": "  📊 Win rate after the edge filter (all qualified): {a:.1f}% (n={n}) | before the filter: "
                       "{b:.1f}% (n={bn})",
    "full_pat_skip": "\n⚠️ pattern discovery skipped: {e}",
    "full_basis_portfolio": "equal-weight portfolio: {p} periods, {t} trades, after a cost of {c}%",
    "full_basis_neutral": " — market-neutral: each trade's return minus the mean of all coins in the same period",
    "full_basis_untimed": "⚠️ no timestamps: sequential trades — not an executable curve",
    "full_tear_title": "📈 Extra trading performance metrics (tearsheet style) — Close target, all coins pooled",
    "full_sig_fail": "⚠️ the significance report failed: {e}",
    "full_done": "✅ The full analysis is complete — executive summary",
    "full_n": "   • samples analysed: {n:,}",
    "full_acc": "   • overall accuracy / hit rate: {a:.2f}%  (Wilson 95%: {lo:.1f}–{hi:.1f}%)",
    "full_cov": "   • coverage: {ev:,} of {tr:,} test rows, {ea} of {ta} coins",
    "full_clipped": "   • uncertainty clipped at NIG_UNC_MAX={m}: {n:,} rows ({p:.2f}%) — out-of-distribution samples",
    "full_ece": "   • ECE before ← after the calibration: {s}",
    "full_trades": "   • trades qualified after the edge filter: {n:,}",
    "full_trust": "   • trust score: {s:.1f}/100",
    "full_decode_ok": "   • decoding verification: ✅ fully sound",
    "full_decode_bad": "   • decoding verification: ❌ {n} failed",
    "full_beats_naive": "   • share of coin / target combinations that really beat the naive baseline: {p:.0f}%",
    "full_sharpe": "   • Sharpe Ratio (Close, {b}): {s:.2f} | Max Drawdown: {d:.1f}%",
    "full_ann": "annualised",
    "full_per_trade": "per trade",
    "full_saved": "\n📁 every big table / figure is saved in: {p}",
}

TEXT = {"ar": AR, "en": EN}


def t(lang: str, key: str, **kw) -> str:
    """The message ``key`` of ``lang`` formatted with ``kw``; an unknown language or key raises."""
    if lang not in TEXT:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    if key not in TEXT[lang]:
        raise KeyError(f"no chicks message {key!r} in {lang!r}")
    return TEXT[lang][key].format(**kw)


class Say:
    """Formats messages in one language: ``say.text(key, **kw)`` returns the string, ``say(key, **kw)`` prints it."""

    def __init__(self, lang: str = "ar"):
        if lang not in TEXT:
            raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
        self.lang = lang

    def text(self, key: str, **kw) -> str:
        return t(self.lang, key, **kw)

    def __call__(self, key: str, **kw) -> None:
        print(self.text(key, **kw))

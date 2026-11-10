# MQM annotation guideline: English → Kumaoni

For native Kumaoni speakers. Read this fully before starting (about 10 minutes).

**Status of this document.** The error dimensions, the three severity levels and the idea of
counting errors come from the MQM framework (themqm.org). Everything marked
**[project]** is an adaptation made for this study and is not part of official MQM.

## 1. Your task

You will see an English sentence and a Kumaoni translation of it. Some translations are
by a machine, some by a person; you are **not told which**. For each pair, mark every
error in the Kumaoni translation, and judge **against the English source**.

You do not see any reference translation. Do not look anything up and do not use any
translation tool. Do not discuss items with the other annotator until the review session.

Aim for about **one minute per item**. Skip nothing; if an item is flawless, mark it so.

## 2. Marking errors

Use one row per error. If an item has three errors, copy the item row twice more and fill
in each error on its own row. If the translation has no errors, write `Y` in `no_error`
and leave the error columns blank.

| Column | What to enter |
|---|---|
| `dimension`, `subtype` | From the table in section 3. |
| `severity` | `neutral`, `minor`, `major` or `critical` (section 4). |
| `span` | The exact words in the translation that are wrong. For an omission, write `[MISSING]` and name the missing English content in `explanation`. |
| `explanation` | One short sentence: why is this an error? |
| `correction` | How you would write it (optional but helpful). Write it in the same script as the item. |
| `annotator_id` | Your ID (already filled in). |

If a single span has two problems, record the more serious one. Only the 5 most
serious errors in an item are counted, so do not spend time listing a sixth.

## 3. Error dimensions

| Dimension | Use when | Subtypes |
|---|---|---|
| `accuracy` | The meaning differs from the English | `addition`, `omission`, `mistranslation`, `untranslated` |
| `linguistic_conventions` | The Kumaoni itself is wrong | `grammar`, `spelling`, `punctuation`, `register`, `inconsistency`, `spelling_variant` |
| `terminology` | A term is wrong for the context | `inappropriate_for_context`, `inconsistent_use` |
| `style` | Correct but unnatural | `awkward` |
| `audience_appropriateness` | Valid, but not a Kumaoni variety you accept | `dialect_mismatch` **[project]** |
| `language_identity` **[project]** | Words from another language | `hindi_leakage`, `other_language` |
| `special` | Whole item is broken, or the source is | `non_translation`, `source_error` |
| `other` | Nothing above fits | `other` |

When two categories fit, choose `accuracy` if the meaning is affected, otherwise the more
specific one.

## 4. Severity

Definitions are paraphrased from MQM; the Kumaoni notes are **[project]**.

- **neutral**: Differs from what you would write, but is acceptable. Not penalised. Record it only
  if it helps the research; otherwise skip it.
- **minor**: A real error with limited impact. A Kumaoni speaker would still understand the
  intended meaning without effort.
- **major**: Seriously affects understanding, meaning or reliability. A reader could be confused
  or misled.
- **critical**: Makes the sentence unfit for use or risky. Examples: reverses the meaning
  (for instance dropping a negation), invents a fact that could cause harm, or the output is not
  Kumaoni at all.

**If you cannot decide between two severities, choose the lower one and say why in the
`explanation`. [project]**

## 5. Kumaoni-specific rules [project]

1. **Spelling variation.** Kumaoni has no single standard spelling in this data (for
   example `chhan / chha / chhe`). If a form is a normal variant, mark it `spelling_variant` with
   severity `neutral`. Use `spelling` only for forms you would call wrong.
2. **Dialect.** If a word or ending is correct in your variety but different from what you expect,
   choose `neutral`. Use `dialect_mismatch` only when you reject it for your variety, and mark
   `minor` unless it impedes understanding. The researcher records your region and variety of
   Kumaoni separately before you start.
3. **Hindi leakage.** Mark `hindi_leakage` for a Hindi word or form where Kumaoni has its own.
   Do **not** mark borrowings that Kumaoni speakers ordinarily use. If a whole sentence is
   effectively Hindi, record one `critical` `hindi_leakage` error on the whole sentence instead of many
   small ones.
4. **Script.** Items appear in Roman script. Do not mark the script itself as an error.
5. **Names and numbers.** A changed name or number is an `accuracy` `mistranslation`; severity is
   `major`, or `critical` if it could mislead.

## 6. Process

1. Briefing and pilot: 10 items together with the researcher; ask questions.
2. Independent annotation: your own sheet, own pace.
3. Review session: the researcher shows only items where you and the other annotator disagree on
   severity by one or more levels. You may discuss and revise. Original annotations are kept.

## 7. Fairness

You are judging the translation, not the person or system. Nothing you mark affects you.

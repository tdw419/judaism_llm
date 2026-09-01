# RAG Evaluation Report

## Overview

- **Total Queries:** 5
- **Successful Queries:** 5
- **Success Rate:** 100.0%
- **Source Citation Accuracy:** 76.7%
- **Average Response Length:** 1757 characters

## Test Results

| Query | Response Length | Sources | Expected Sources | Match |
|-------|----------------|---------|------------------|-------|
| What is Teshuva? | 1856 | Orot HaTeshuvah, Orot HaTeshuvah, Tikkun HaKlali, Metzudat David on I Samuel, Pardes Rimmonim | Mishneh Torah, Talmud | ✗ |
| Explain Shabbat | 2461 | Bartenura on Mishnah Beitzah, English Explanation of Mishnah Shabbat, Shulchan Shel Arba, Benei Binyamin on Mishneh Torah, Shofar, Sukkah and Lulav, Jerusalem Talmud Beitzah | Torah, Mishnah | ✓ |
| What are the main sources of Jewish law? | 1548 | Conversion "According to Halakhah"; What Is It, Mishneh Torah, Transmission of the Oral Law, Mishnat Eretz Yisrael on Pirkei Avot, Minchat Chinukh, English Explanation of Pirkei Avot | Torah, Mishnah, Talmud | ✓ |
| תורה | 2241 | Vayikra Rabbah, Boaz on Mishnah Eduyot, Chayei Moharan, Mishneh Torah, Vessels, Piskei Tosafot on Moed Katan | Torah | ✓ |
| משנה | 677 | Vayikra Rabbah, Boaz on Mishnah Eduyot, Mishneh Torah, Vessels, Notes by Rabbi Yehoshua Hartman on Derush al HaTorah, Chayei Moharan | Mishnah | ✓ |

## Sample Responses

### Query 1: What is Teshuva?

Teshuva, repentance, is a fundamental concept in Jewish thought and practice, involving not only the regret over past actions but also a commitment to change one's behavior for the better. According to Rabbi Avraham Yitzchak Kook (רבי אברהם יצחק קוק), the first principle of Teshuva is to return to God and to mend one's relationship with Him through sincere repentance. This process involves acknowledging one’s sins, regretting them, and making a firm resolve not to repeat them. Here is a relevant excerpt from his work:

From Orot HaTeshuvah: "The essence of teshuva is the restoration of the soul to its original state, which is to say, its original connection with God. The soul was created in purity and holiness, but through sin it has become defiled and estranged from its Creator. Teshuva is the means by which the soul can be cleansed and returned to its pristine condition." [Orot HaTeshuvah_1_1]

Additionally, Rabbi Kook emphasizes that true teshuva involves not just external behavior but also internal transformation: "Repentance must encompass both the body and the soul, the inner and the outer. It is not sufficient merely to refrain from transgressing; one must actively engage in positive actions and spiritual growth." [Orot HaTeshuvah_1_0]

This transformative process is further elaborated upon in his commentary on various biblical passages and ethical teachings, underscoring the holistic nature of repentance. For instance, in his commentary on I Samuel 7:6, Rabbi Kook discusses the importance of communal repentance and the collective responsibility of the community to return to God together. [Metzudat David on I Samuel_1_0]

Thus, according to Rabbi Kook, teshuva involves a comprehensive return to God, encompassing both individual and communal dimensions, and is a continuous process of spiritual purification and renewal.


## Conclusions

✓ RAG system performs well (success rate > 80%)
✓ Source citations are accurate (accuracy > 70%)

## Recommendations

1. **If success rate is low:**
   - Increase top_k in retrieval
   - Improve embedding model
   - Expand Sefaria corpus

2. **If source accuracy is low:**
   - Improve metadata quality
   - Add more source references
   - Refine citation extraction

3. **If responses are too short:**
   - Increase max_new_tokens
   - Improve context assembly
   - Fine-tune model further

---

**Generated:** September 1, 2026
**Model:** judaism-llm-qwen2.5-7b-merged
**Corpus:** 18,453 Sefaria segments

"""Generate the Mandarin data tables for the Kokoro backend.

The Kokoro v1.0 Mandarin voices were trained on phonemes from misaki's
legacy Chinese G2P: jieba words -> pypinyin (TONE3) -> pinyin-to-IPA. This
script snapshots the lookup data that pipeline uses, so the Rust port
produces the same phonemes without any Python at runtime.

    python3 -m venv /tmp/zhgen && /tmp/zhgen/bin/pip install "misaki[zh]==0.9.4" "pypinyin==0.55.0"
    /tmp/zhgen/bin/python scripts/gen_kokoro_zh_data.py

Outputs (in crates/wavekat-tts/src/backends/kokoro/zh/):
  chars.tsv      char <TAB> default reading, for U+4E00..U+9FFF
  phrases.tsv    phrase [<TAB> readings], every pypinyin phrase; readings
                 only where they differ from char-by-char defaults
  syllables.tsv  reading <TAB> Kokoro phonemes
  golden.tsv     sentence <TAB> phonemes from misaki itself, for tests

Data sources: pypinyin (MIT), whose tables come from mozillazg/pinyin-data
and mozillazg/phrase-pinyin-data (MIT); the syllable mapping comes from
misaki (Apache-2.0), adapted from stefantaubert/pinyin-to-ipa (MIT).
"""

from pathlib import Path

from misaki.zh import ZHG2P
from pypinyin.contrib.tone_convert import to_tone3
from pypinyin.phrases_dict import phrases_dict
from pypinyin.pinyin_dict import pinyin_dict

OUT = Path(__file__).resolve().parent.parent / "crates/wavekat-tts/src/backends/kokoro/zh"


def is_han(c: str) -> bool:
    return "一" <= c <= "鿿"


def tone3(py: str) -> str:
    # lazy_pinyin(style=TONE3, neutral_tone_with_five=True) spelling.
    return to_tone3(py, neutral_tone_with_five=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    chars = {
        chr(code): tone3(readings.split(",")[0])
        for code, readings in sorted(pinyin_dict.items())
        if is_han(chr(code))
    }

    phrases = {}
    for word, readings in sorted(phrases_dict.items()):
        if len(word) < 2 or not all(c in chars for c in word):
            continue
        r = [tone3(p[0]) for p in readings]
        if len(r) != len(word):
            continue
        phrases[word] = "" if r == [chars[c] for c in word] else " ".join(r)

    syllables = {}
    for r in sorted(set(chars.values()) | {s for v in phrases.values() for s in v.split()}):
        try:
            ipa = ZHG2P.py2ipa(r).replace(chr(815), "")
        except Exception:
            continue  # misaki can't read it either; the char is skipped
        syllables[r] = ipa

    with open(OUT / "chars.tsv", "w", encoding="utf-8") as f:
        for c, r in chars.items():
            f.write(f"{c}\t{r}\n")
    with open(OUT / "phrases.tsv", "w", encoding="utf-8") as f:
        for w, r in phrases.items():
            f.write(f"{w}\t{r}\n" if r else f"{w}\n")
    with open(OUT / "syllables.tsv", "w", encoding="utf-8") as f:
        for r, ipa in syllables.items():
            f.write(f"{r}\t{ipa}\n")

    with open(OUT / "golden.tsv", "w", encoding="utf-8") as f:
        for s in GOLDEN:
            ps = ZHG2P.legacy_call(ZHG2P.map_punctuation(s))
            f.write(f"{s}\t{ps}\n")

    print(f"{len(chars)} chars, {len(phrases)} phrases, {len(syllables)} syllables")


# Pure Chinese (no digits or Latin) so misaki's legacy path is the reference.
GOLDEN = [
    "您好，您的订单已经确认，我们会尽快为您安排发货。",
    "提醒您，明天下午三点开会，请按一确认，按二取消。",
    "系统通知：您的账户密码已修改。如果不是您本人操作，请立即联系客服。",
    "我们还是要去重庆，银行的工作很重要。",
    "长大以后，他成了一名音乐老师。",
    "这个问题不好回答，我们需要再讨论一下。",
    "请问您是需要转接人工服务吗？",
    "您的快递已经到达小区门口的快递柜，取件码已发送到您的手机。",
    "会议改到星期五上午十点，地点在三楼大会议室。",
    "感谢您的来电，祝您生活愉快！",
    "抱歉，我没有听清楚，请您再说一遍。",
    "他把行李放在了银行门口。",
    "这首歌的旋律非常好听，大家都喜欢唱。",
    "我觉得这件事情还需要再考虑考虑。",
    "绿色的树叶在风中轻轻摇动。",
    "《人民日报》今天发表了一篇文章。",
    "请输入您的密码，然后按井号键结束。",
    "如果需要帮助，请随时联系我们。",
]

if __name__ == "__main__":
    main()

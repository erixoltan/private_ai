import unittest

from private_ai.terminal import WordWrapper


def wrap(pieces, width):
    out = []
    w = WordWrapper(out.append, lambda: width)
    for piece in pieces:
        w.feed(piece)
    w.flush()
    return "".join(out)


class WordWrapperTest(unittest.TestCase):
    def test_words_split_across_pieces_are_not_broken(self):
        self.assertEqual(wrap(["The qui", "ck brown f", "ox jumps"], 10), "The quick\nbrown fox\njumps")

    def test_no_line_exceeds_width(self):
        text = "one two three four five six seven eight nine ten " * 5
        for line in wrap(list(text), 17).split("\n"):
            self.assertLessEqual(len(line), 17)

    def test_newlines_and_blank_lines_are_kept(self):
        self.assertEqual(wrap(["a b\n\nc d\n"], 80), "a b\n\nc d\n")

    def test_wrapped_lines_keep_indentation(self):
        self.assertEqual(wrap(["  - alpha beta gamma\nnext"], 12), "  - alpha\n    beta\n    gamma\nnext")
        self.assertEqual(wrap(["  plain alpha beta\nnext"], 12), "  plain\n  alpha beta\nnext")
        self.assertEqual(wrap(["10. alpha beta gamma"], 14), "10. alpha beta\n    gamma")

    def test_code_indentation_is_untouched(self):
        self.assertEqual(wrap(["    x = 1\n        y = 2\n"], 80), "    x = 1\n        y = 2\n")

    def test_overlong_word_is_left_whole(self):
        self.assertEqual(wrap(["see https://example.com/a/very/long/path ok"], 10),
                         "see\nhttps://example.com/a/very/long/path\nok")

    def test_wide_characters_count_double(self):
        self.assertEqual(wrap(["\u65e5\u672c \u8a9e\u8a9e ab"], 6), "\u65e5\u672c\n\u8a9e\u8a9e\nab")

    def test_width_change_applies_to_later_text(self):
        out, width = [], [20]
        w = WordWrapper(out.append, lambda: width[0])
        w.feed("aaa bbb ccc ")
        width[0] = 7
        w.feed("ddd eee")
        w.flush()
        self.assertEqual("".join(out), "aaa bbb ccc\nddd eee")


if __name__ == "__main__":
    unittest.main()

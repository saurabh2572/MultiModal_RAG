import unittest

from multimodal_rag.chunker import chunk_elements


BASE = {
    "source": "report.pdf", "source_path": "report.pdf", "source_sha256": "abc",
    "page": 12, "page_label": "8", "page_width": 595, "page_height": 842,
    "section": "Financial Statements", "bbox": [10, 10, 200, 80],
    "extraction_method": "pymupdf_native",
}


class ChunkerTests(unittest.TestCase):
    def test_text_blocks_group_on_page_and_keep_provenance(self):
        elements = [
            BASE | {"id": "a", "modality": "text", "text": "Revenue rose during the year."},
            BASE | {"id": "b", "modality": "text", "text": "Profit followed the same trend."},
        ]
        chunks = list(chunk_elements(elements))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["element_ids"], ["a", "b"])
        self.assertIn("Profit followed", chunks[0]["text"])

    def test_long_table_preserves_every_row_with_repeated_context(self):
        rows = [[f"Item {i}", "100" * 15] for i in range(1, 8)]
        table = BASE | {"id": "table", "modality": "table", "rows": rows,
                        "context_above": "Amount in crore", "table_index": 1}
        chunks = list(chunk_elements([table], max_tokens=100, overlap_tokens=10))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["row_start"], 1)
        self.assertEqual(chunks[-1]["metadata"]["row_end"], len(rows))
        for chunk in chunks:
            self.assertIn("Amount in crore", chunk["text"])
        for i in range(1, 8):
            self.assertEqual(sum(f"Item {i} |" in chunk["text"] for chunk in chunks), 1)


if __name__ == "__main__":
    unittest.main()

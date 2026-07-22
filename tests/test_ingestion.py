import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / "vendor"))

from lyy_rag_agent.ingestion import IngestionPipeline  # noqa: E402
from lyy_rag_agent.loaders import LoaderError, LoaderFactory, PdfLoader  # noqa: E402
from lyy_rag_agent.security import SecurityAgent  # noqa: E402


class IngestionTest(unittest.TestCase):
    def test_text_import_is_sanitized_and_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text(
                "客户邮箱 demo@example.com，ASIN B012345678，账号 123456789，电话 13800138000",
                encoding="utf-8",
            )
            output, audit, metadata = IngestionPipeline(root, SecurityAgent()).extract(source)
            text = output.read_text(encoding="utf-8")
            self.assertEqual(metadata["status"], "pending_review")
            self.assertTrue(audit.exists())
            self.assertNotIn("demo@example.com", text)
            self.assertNotIn("B012345678", text)
            self.assertNotIn("13800138000", text)
            self.assertIn("[已脱敏:email]", text)

            with self.assertRaises(ValueError):
                IngestionPipeline(root, SecurityAgent()).approve(
                    output, "标题", "分类", "主题", confirmed=False
                )
            target = IngestionPipeline(root, SecurityAgent()).approve(
                output, "已审核标题", "配置规范", "导入测试", confirmed=True
            )
            self.assertTrue(target.exists())
            self.assertIn("status: reviewed", target.read_text(encoding="utf-8"))

    def test_minimal_docx_and_pptx_loaders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docx = root / "sample.docx"
            with zipfile.ZipFile(str(docx), "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    '<w:document xmlns:w="urn:w"><w:body><w:p><w:r><w:t>Word 内容</w:t></w:r></w:p></w:body></w:document>',
                )
            self.assertIn("Word 内容", LoaderFactory.load(docx).text)

            pptx = root / "sample.pptx"
            with zipfile.ZipFile(str(pptx), "w") as archive:
                archive.writestr(
                    "ppt/slides/slide1.xml",
                    '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>PPT 内容</a:t></p:sld>',
                )
            self.assertIn("PPT 内容", LoaderFactory.load(pptx).text)

    def test_pdf_page_spec(self):
        self.assertEqual(PdfLoader._page_numbers("2-4,6", 8), [2, 3, 4, 6])
        with self.assertRaises(LoaderError):
            PdfLoader._page_numbers("9", 8)


if __name__ == "__main__":
    unittest.main()

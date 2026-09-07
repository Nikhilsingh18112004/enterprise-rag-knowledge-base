#!/usr/bin/env python3
"""
Simple PDF text extractor using pypdf.

This script accepts a PDF file path, extracts text page-by-page,
and prints each page's text to standard output.
"""

# Import modules from the standard library.
import argparse  # For parsing command-line arguments.
from pathlib import Path  # For convenient filesystem path handling.
import sys  # For exiting with error codes and writing to stderr.

# Import PdfReader from the pypdf package.
# Install with: pip install pypdf
from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path):
	"""Extract text from each page of a PDF.

	Args:
		pdf_path (Path): Path object pointing to the PDF file.

	Returns:
		List[tuple[int, str]]: A list of tuples where each tuple is
		(page_number, extracted_text_for_that_page).
	"""
	# Create a PdfReader instance which opens and parses the PDF file.
	reader = PdfReader(str(pdf_path))

	# Prepare a list to hold (page_number, text) pairs.
	pages_text = []

	# Iterate over the document's pages. enumerate gives us page index.
	for i, page in enumerate(reader.pages, start=1):
		# Extract the text for the current page. extract_text() returns
		# a string or None if nothing was extracted, so fall back to an
		# empty string to avoid printing 'None'.
		text = page.extract_text() or ""

		# Append a tuple containing the 1-based page number and the text.
		pages_text.append((i, text))

	# Return the list of extracted page texts to the caller.
	return pages_text


def main():
	# Create an ArgumentParser to handle command-line arguments.
	parser = argparse.ArgumentParser(
		description="Extract text from a PDF page-by-page and print it."
	)

	# Add a positional argument 'pdf' which is the path to the PDF file.
	parser.add_argument("pdf", help="Path to the PDF file to extract text from")

	# Parse the arguments provided by the user when running the script.
	args = parser.parse_args()

	# Convert the provided path string to a Path object for easier checks.
	pdf_path = Path(args.pdf)

	# Verify that the file exists before attempting to read it.
	if not pdf_path.exists():
		# Print an error message to stderr and exit with a non-zero status.
		print(f"Error: file not found: {pdf_path}", file=sys.stderr)
		sys.exit(1)

	# Try to extract text from the PDF, and handle any exceptions
	# that might be raised by pypdf for malformed or encrypted files.
	try:
		pages = extract_pdf_text(pdf_path)
	except Exception as exc:
		print(f"Failed to read PDF: {exc}", file=sys.stderr)
		sys.exit(1)

	# Print the extracted text page-by-page with a simple header.
	for page_number, text in pages:
		# Print a visual separator and the page number.
		print(f"\n=== Page {page_number} ===\n")

		# Print the text for the page. If the page had no text this prints
		# an empty line which is fine for clarity.
		print(text)


if __name__ == "__main__":
	# If the script is executed directly (not imported), run main().
	main()


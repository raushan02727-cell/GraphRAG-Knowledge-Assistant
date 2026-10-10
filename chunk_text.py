import fitz

pdf_path = r"C:\Users\raush\Downloads\the-son-of-the-wolf-jack-london-1150.pdf"
document = fitz.open(pdf_path)

# Read all PDF text
full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()

# Split text into chunks
chunk_size = 5000

chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)

print("Total chunks:", len(chunks))

# Show first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print("\n==========================")
    print("CHUNK", i + 1)
    print("==========================")
    print(chunk)
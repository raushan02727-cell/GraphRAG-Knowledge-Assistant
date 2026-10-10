import fitz

pdf_path =  r"C:\Users\raush\Downloads\the-son-of-the-wolf-jack-london-1150.pdf"
document = fitz.open(pdf_path)

print("Number of pages:", len(document))

for page_number, page in enumerate(document):
    text = page.get_text()

    print("\n-----------------------------")
    print("PAGE:", page_number + 1)
    print("-----------------------------")

    print(text[:5000])

document.close()
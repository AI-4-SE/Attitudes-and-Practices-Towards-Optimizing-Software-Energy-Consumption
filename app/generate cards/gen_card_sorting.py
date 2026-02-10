import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from textwrap import wrap

# Load the CSV
df = pd.read_csv('/home/mweber/projects/energy consumption online survey/results-survey762399.csv')  # Replace with actual file path

def gen_card(qid, pid, question, answer, c, x, y, card_width, card_height):

    # Header line
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x + 5, y + card_height - 12, qid)
    c.drawRightString(x + card_width - 5, y + card_height - 12, pid)

    # Question
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x + 5, y + card_height - 25, "Question:")
    c.setFont("Helvetica", 6)
    wrapped_question = wrap(question, width=95)
    text_obj = c.beginText(x + 5, y + card_height - 33)
    for line in wrapped_question:
        text_obj.textLine(line)
    c.drawText(text_obj)

    # Answer
    c.setFont("Helvetica-Bold", 8)
    text_y = y + card_height - 30 - (len(wrapped_question) * 10)
    if text_y < y + 30:
        text_y = y + 30  # minimum spacing
    c.drawString(x + 5, text_y, "Answer:")
    c.setFont("Helvetica", 8)
    wrapped_answer = wrap(answer, width=70)

    # special handling for long answers
    if len(answer) > 800:
        c.setFont("Helvetica", 7)
        wrapped_answer = wrap(answer, width=85)
    if  len(answer) > 1100:
        c.setFont("Helvetica", 6)
        wrapped_answer = wrap(answer, width=100)
    if  len(answer) > 1500:
        c.setFont("Helvetica", 5)
        wrapped_answer = wrap(answer, width=115)

    text_obj = c.beginText(x + 5, text_y - 12)
    for line in wrapped_answer:
        text_obj.textLine(line)
    c.drawText(text_obj)

# Page setup
page_width, page_height = A4
cols = 2
rows = 4
margin_x = 6 * mm
margin_y = 6 * mm
inner_spacing_x = 2 * mm
inner_spacing_y = 2 * mm

# Calculate available width/height for cards
usable_width = page_width - 2 * margin_x - (cols - 1) * inner_spacing_x
usable_height = page_height - 2 * margin_y - (rows - 1) * inner_spacing_y
card_width = usable_width / cols
card_height = usable_height / rows

# PDF setup
c = canvas.Canvas("cards.pdf", pagesize=A4)


question_map = {
    'RQ2': 'Aus Ihrer Sicht, lohnt sich der Aufwand, Energieverbrauch durch Optimierung von Software zu reduzieren? Bitte erläutern Sie.',
    'ECP4': 'Auf welche Faktoren (wie Cloud-Kosten, Laufzeit, CPU-Auslastung, Batterieverbrauch, auswahl effizienter Hardware usw.) konzentrieren Sie sich beim optimieren Ihrer Software? Zählen sie auf.',
    'RE2': 'Bitte beschreiben Sie Gemeinsamkeiten und Unterschiede beim Debuggen von Energieproblemen gegenüber anderen Fehlern, wie performance- und funktionalen Fehlern:',
    'EnergyProxy01': 'Welche Gründe gibt es für Sie und Ihr Unternehmen, den Energieverbrauch nicht direkt zu messen?',
    'EnergyProxy02': 'Basierend auf Ihren bisherigen Antworten: Sie beurteilen den Energieverbrauch basierend auf direkten Messungen als auch durch indirekte Messungen. Warum nutzen Sie neben direkten Energiemessungen auch indirekte Messungen?',
    'ProxyMerge01': 'Glauben Sie, dass es einen Zusammenhang zwischen Laufzeit (Performance) und Energieverbrauch gibt? Bitte erläutern Sie Ihre Ansichten, insbesondere Ihre Erwartungen, inwieweit Energieverbrauch und Leistung voneinander abhängig sind oder nicht.',
    'CO0': 'Wenn Sie an Ihre vorherigen Antworten denken: Was sind die größten Probleme und Herausforderungen, die noch überwunden werden müssen, damit der Energieverbrauch von Software eine größere Rolle spielt?',
    'CO1': 'Was muss Ihrer Meinung nach getan werden, um die Herausforderungen des Energieverbrauchs zu bewältigen?',
    'CQ3': 'Glauben Sie, dass es mehr Regulierungen (z. B. Gesetze) oder politische Anreize (z. B. spezielle Förderungen) braucht, um die Praxisrelevanz des Energieverbrauchs von Softwareprodukten zu erhöhen? Bitte erläutern Sie Ihre Sichtweise.',
    'CQ0': 'Gibt es noch etwas, dass Sie uns bezüglich des Energieverbrauchs von Software oder allgemein zu dieser Umfrage mitteilen möchten?'
}

i = 0

# Iterate over participants
for _, row in df.iterrows():

    # Iterate over questions
    for question_key, question_value in question_map.items():
        if str(row[question_key]) != 'nan':

            col = i % cols
            row_pos = (i // cols) % rows
            page = i // (cols * rows)

            # Start a new page every 8 cards
            if i > 0 and i % (cols * rows) == 0:
                c.showPage()

            # Calculate card position
            x = margin_x + col * (card_width + inner_spacing_x)
            y = page_height - margin_y - (row_pos + 1) * (card_height + inner_spacing_y)

            qid = f"#QID {question_key}"
            pid = f"#PID {row['id']}"
            question = question_value
            answer = row[question_key]
        
            # Draw border
            c.rect(x, y, card_width, card_height)
            gen_card(qid, pid, question, answer, c, x, y, card_width, card_height)

            # Increment CARD if we print a card
            i += 1

c.save()

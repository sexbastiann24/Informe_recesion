"""
Genera el correo del reporte de probabilidad de recesión con Groq.

Lee daily_summary.json, se lo pasa a Qwen 3.8 27B (con hasta 3 gráficos,
que es el máximo de imágenes por consulta de ese modelo) y arma el HTML.
El envío por Gmail es opcional.

Uso:
    python automation/generate_email.py

Variables de entorno (también se leen de ../.env):
    GROQ_API_KEY   - clave de console.groq.com
    MAIL_USERNAME  - Gmail (opcional, para enviar)
    MAIL_PASSWORD  - contraseña de aplicación de Gmail (opcional)
    MAIL_PORT      - puerto SMTP, normalmente 587
    EMAIL_TO       - destinatarios, separados por coma
"""

import os
import sys
import json
import base64
from datetime import datetime
from pathlib import Path

from groq import Groq

OUTPUT_DIR = Path(__file__).parent / "output"
MODELO = "qwen/qwen3.8-27b"
# El plan gratuito de Groq admite 7 000 tokens de entrada por minuto y cada
# imagen cuenta como 2 048. El JSON del reporte ya se acerca a ese tope,
# así que los gráficos no se envían al modelo: se incrustan después en el correo.
MAX_IMAGENES = 0


def cargar_env():
    """Carga el .env de la raíz del repo sin pisar variables ya definidas."""
    ruta = Path(__file__).resolve().parents[1] / ".env"
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        texto = linea.strip()
        if not texto or texto.startswith("#") or "=" not in texto:
            continue
        clave, valor = texto.split("=", 1)
        clave, valor = clave.strip(), valor.strip().strip('"')
        if valor and clave not in os.environ:
            os.environ[clave] = valor


cargar_env()


def load_summary():
    """Load the daily summary JSON."""
    path = OUTPUT_DIR / "daily_summary.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run daily_report.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def encode_image(path):
    """Base64-encode an image for Claude's vision API."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def generate_analysis(summary):
    """Envía el resumen y hasta 3 gráficos a Qwen para redactar el correo."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Build the prompt
    run_date = summary.get("run_date", datetime.now().strftime("%Y-%m-%d"))
    data_date = summary.get("data_through", "unknown")

    prompt = f"""Eres un estratega macro senior. Redacta en español un informe semanal
de probabilidad de recesión para un comité de inversión.

FECHA DE HOY: {run_date}
DATOS HASTA: {data_date}

REGLA DE IDIOMA: todo el texto visible va en español. Los códigos de variable
se escriben exactamente como vienen en el JSON. No los traduzcas ni los sustituyas
por un nombre largo. Ejemplos que deben quedar literales: SPREAD, FEDFUNDS,
HSN1F_YOY, CPILFESL_YOY, PPIACO_YOY, HOUST_YOY, UNRATE_CHG3. Los nombres de
modelos sí se escriben en español: «NY Fed (solo SPREAD)», «Wright (SPREAD + FF)»,
«Seleccionado por BIC», «Estrella-Mishkin» y «Chauvet-Piger».

FORMATO: debajo del título y de la fecha, pon en un lugar visible la línea
«Datos hasta: {data_date}». No la escondas en letra pequeña.

Toda referencia temporal se ancla a la fecha de hoy ({run_date}). No menciones
meses que todavía no han ocurrido. Lo que haya que vigilar se refiere a la
próxima publicación de datos respecto de {run_date}.

Con la salida del modelo, redacta un informe sustantivo, con los gráficos
integrados en el texto.

Hay 7 gráficos. Referéncialos en el HTML con <img src="cid:chart_0"> hasta
<img src="cid:chart_6">. El orden es:
- cid:chart_0 = medidor de la probabilidad actual
- cid:chart_1 = tendencia a 24 meses
- cid:chart_2 = percentil histórico de cada indicador
- cid:chart_3 = comparación de modelos
- cid:chart_4 = trayectoria de cada indicador, 24 meses
- cid:chart_5 = historia completa, con sombras de recesión NBER
- cid:chart_6 = sensibilidad y umbrales

Inserta cada gráfico dentro de la sección que ilustra, justo después del
párrafo correspondiente. No los agrupes al principio ni al final. Debajo de
cada uno, un pie de foto breve en gris y en español.

SALIDA DEL MODELO:
{json.dumps(summary, indent=2)}

Escribe el correo con estas secciones:
1. **Asunto**, una línea, con este formato: «Probabilidad de recesión: [ensamble]% —
   los modelos van de [mín]% a [máx]%». No uses etiquetas de semáforo
   (LOW, MODERATE, HIGH, bajo, moderado, alto) ni en el asunto ni en el texto.
   El medidor de colores ya clasifica visualmente.

2. **Resumen para el comité.** Es la primera sección, justo después de
   «Datos hasta». Cabe en una página y lleva estos bloques, en este orden:

   TABLA (4 filas, en HTML). Los rótulos van en español:
   | Probabilidad actual | [ensamble]% |
   | Dirección del cambio | [al alza / a la baja / estable, según la tendencia] |
   | Principal factor de riesgo | [el indicador más cerca de su umbral de alerta; conserva su código] |
   | Principal factor de alivio | [el indicador que más sostiene la expansión; conserva su código] |

   RANGO DE MODELOS, una frase: «Los modelos individuales van de [mín]% a [máx]%,
   lo que refleja un desacuerdo relevante sobre la lectura de la curva de rendimientos.»

   POSICIONAMIENTO, exactamente 4 viñetas:
   - Renta variable: [postura] — [una frase con la fórmula «consistente con»]
   - Renta fija: [postura] — [una frase]
   - Crédito: [postura] — [una frase]
   - Coberturas: [postura] — [una frase]
   Toda postura se formula como «consistente con [condición]», nunca como una orden.
   Cierra en cursiva: «El posicionamiento refleja solo la salida del modelo y debe
   evaluarse frente a las restricciones de cada mandato.»

   Después de esta sección, chart_0 (el medidor).

3. **Resumen ejecutivo** (3 a 5 frases). La cifra principal es la probabilidad del
   ENSAMBLE. Empieza así: «Nuestro ensamble de cinco modelos estima un [X]% de
   probabilidad de recesión en los próximos 12 meses, consistente con una fase de
   expansión, aunque la dispersión entre modelos merece atención.» No uses
   «riesgo bajo», «riesgo moderado» ni «riesgo alto». Describe con qué es consistente
   ese nivel. Señala la fuerza del consenso. El modelo seleccionado por BIC es uno
   de cinco. Si divergen, dilo con el dato y deja la explicación para la sección
   de divergencia.

   Después, chart_1 (tendencia) y chart_3 (comparación).

4. **Indicadores clave.** Exactamente cuatro bloques. Cada uno, dos frases:
   qué muestran ahora y qué implica eso para el riesgo de recesión. No definas
   qué mide cada indicador. No des contexto histórico.

   **Crecimiento** (HOUST, HSN1F): lectura e implicación.
   **Inflación** (CPILFESL_YOY, PPIACO_YOY): lectura e implicación.
   **Política** (FEDFUNDS, SPREAD): lectura e implicación.
   **Señales de mercado** (diferenciales de crédito, sentimiento): lectura e implicación.

   Después, chart_2 (percentiles) y chart_4 (trayectorias).

5. **Divergencia entre modelos.** Exactamente tres párrafos:

   Párrafo 1. Por qué sigue importando la curva: el diferencial de plazos ha
   precedido cada recesión desde 1968. La fase actual, de salida de la inversión,
   es históricamente la más delicada: las recesiones suelen empezar entre 6 y 18
   meses después de que la curva se empina tras haber estado invertida.

   Párrafo 2. Por qué este ciclo puede ser distinto, solo estos tres factores:
   (a) la expansión cuantitativa comprimió de forma artificial la prima de plazo,
   así que la inversión se alcanza sin un endurecimiento equivalente del crédito;
   (b) la demanda de Treasuries por bancos centrales extranjeros bajó los tipos
   largos al margen de las expectativas de crecimiento; (c) la regulación posterior
   a Basilea III redujo el apetito bancario por duración y aplanó la curva de
   manera estructural.

   Párrafo 3. El juicio: una frase sobre qué lectura implica la ponderación del
   ensamble, y otra sobre qué evidencia la confirmaría o la descartaría.

   Cierra la sección con esta nota, literal: «Nota: el modelo asigna un coeficiente
   negativo a la inflación, porque históricamente las recesiones por colapso de la
   demanda vienen precedidas de desinflación. Eso puede subestimar el riesgo de
   estanflación cuando la inflación y la debilidad del crecimiento coinciden.»

   Después, chart_5 (historia).

6. **Lista de seguimiento.** Tabla con los datos de sensibilidad, ordenada por
   el umbral más cercano al valor actual. En cada fila: el código del indicador,
   el valor actual, el umbral, la distancia y qué hecho real podría provocar el
   movimiento. Máximo dos frases por indicador.

   Después, chart_6.

7. **Escenario adverso.** No le asignes una probabilidad numérica: el modelo no
   estima probabilidades conjuntas condicionales, e inventar una sería engañoso.

   CLASIFICACIÓN, literal: «Riesgo de cola. Exige un deterioro simultáneo de
   indicadores poco correlacionados, algo históricamente raro fuera de una crisis
   financiera sistémica o de un shock externo.»

   DISPARADORES, tres viñetas:
   (a) Error grave de la Fed: endurecer de más con el crecimiento ya flojo y verse
       forzada a un giro brusco que desordene el crédito.
   (b) Shock energético: petróleo por encima de 130 dólares de forma sostenida,
       que reacelere PPIACO_YOY y frene el consumo.
   (c) Evento de crédito: tensión en bancos regionales o contagio soberano, con
       ampliación de diferenciales y contracción del crédito.

   IMPLICACIÓN PARA COBERTURAS, un párrafo, aplicado a la lectura ACTUAL del ensamble:
   - Por debajo de 20% y tendencia estable: solo vigilancia y rebalanceo habitual.
   - Entre 20% y 35% con tendencia al alza: coberturas de cola (volatilidad larga,
     más duración en Treasuries).
   - Por encima de 35%: conviene un reposicionamiento defensivo.
   Di la lectura actual, en qué tramo cae y la conclusión explícita.

   QUÉ VIGILAR: dos frases sobre cuál de los tres disparadores está más cerca
   con los datos de hoy.

8. **Qué cambiaría la lectura.** Lista numerada de exactamente cinco puntos.
   Usa estos textos, sin recalcular los umbrales ni cambiar el sentido:

   1. SPREAD cae por debajo de -1,0%: históricamente asociado a un aterrizaje brusco;
      obligaría a revisar el ensamble al alza de forma material.
   2. La inflación subyacente (CPILFESL_YOY) baja de 1,5%: la destrucción de demanda
      va por delante de la normalización de la oferta; un riesgo de deflación
      incompatible con un aterrizaje suave.
   3. HOUST cae más de un 15% interanual: la transmisión de la tasa hipotecaria se
      acelera más allá de la fase de estabilización.
   4. Las solicitudes iniciales de subsidio de desempleo se sostienen por encima de
      300.000 en el promedio de cuatro semanas: el empleo se deteriora antes de que
      reaccione la tasa de desempleo.
   5. El ensamble supera el 20% durante dos actualizaciones mensuales seguidas:
      señal de convergencia que justificaría revisar un reposicionamiento defensivo.

9. **Vigencia de los datos.** Una caja gris con borde. Usa los campos
   "data_through" y "lagged_series" del JSON, pero redacta en español:
   «Aviso de vigencia: este informe usa datos de FRED disponibles hasta
   [data_through]. Las series con rezago de publicación superior a 30 días,
   que se actualizarán en su próxima publicación, son: [lista, o «ninguna»].
   Las condiciones pueden haber cambiado desde ese corte. Las probabilidades
   se actualizan en la próxima corrida programada.»

10. **Cierre para el comité.** 2 o 3 frases. Empieza con el ensamble:
    «El ensamble de cinco modelos está en [X]%, consistente con condiciones de
    [expansión / contracción / transición].» Menciona el umbral más cercano de
    la lista de seguimiento y di qué cambiaría la recomendación. Sin etiquetas
    de semáforo.

REGLAS DE REDACCIÓN:
- No repitas lo que el gráfico ya muestra.
- No definas qué mide un indicador.
- No uses «cabe destacar», «es importante señalar» ni «vale la pena notar».
- No cubras cada frase con un condicional. Escribe con convicción.
- Nunca describas la probabilidad como «riesgo bajo» o «riesgo alto». Di con
  qué situación es consistente.

El cuerpo va en HTML limpio, con CSS en línea, apto para un cliente de correo.
Extensión: 1000 a 1500 palabras. Es un memorando institucional, no un resumen
de blog. Tono de un economista senior ante la dirección: concreto y con criterio.
Sin emojis. Usa el signo de porcentaje.

Responde en JSON con dos claves:
- "subject": el asunto, en español
- "html_body": el cuerpo HTML completo, en español
"""

    # Qwen acepta como máximo 3 imágenes por consulta. El JSON ya trae
    # todos los números, así que el resto de los gráficos solo se incrusta
    # después, en el correo.
    content = [{"type": "text", "text": prompt}]
    disponibles = [n for n in summary.get("charts", []) if (OUTPUT_DIR / n).exists()]
    for chart_name in disponibles[:MAX_IMAGENES]:
        img_data = encode_image(OUTPUT_DIR / chart_name)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_data}"},
        })
        content.append({
            "type": "text",
            "text": f"[Above image: {chart_name}]",
        })
    if len(disponibles) > MAX_IMAGENES:
        print(f"Se envían {MAX_IMAGENES} de {len(disponibles)} gráficos al modelo.")

    print(f"Sending to Groq ({MODELO}) for analysis...")
    response = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODELO,
                max_completion_tokens=8192,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
            )
            break
        except Exception as e:
            if attempt < 3:
                import time
                wait = 2 ** (attempt + 1)
                print(f"  API error (attempt {attempt+1}/4): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API failed after 4 attempts: {e}")
                raise

    # Parse response
    response_text = response.choices[0].message.content or ""

    # Try to extract JSON from the response
    try:
        # Handle case where Claude wraps JSON in markdown code blocks
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text
        result = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: use the raw text as the body
        result = {
            "subject": f"Probabilidad de recesión — {summary['bic_probability']}%",
            "html_body": f"<html><body><pre>{response_text}</pre></body></html>",
        }

    return result


def embed_charts_in_html(html_body, summary):
    """Replace chart references with CID-based inline images for email."""
    for i, chart_name in enumerate(summary.get("charts", [])):
        cid = f"chart_{i}"
        # Add image tags where appropriate (at the end if not already present)
        if chart_name not in html_body:
            section_map = {
                "recession_probability_gauge.png": "Resumen para el comité",
                "recession_probability_history.png": "Divergencia entre modelos",
                "sensitivity_chart.png": "Lista de seguimiento",
            }
            for keyword in section_map.values():
                if keyword.lower() in html_body.lower():
                    # Insert image after the relevant section
                    idx = html_body.lower().find(keyword.lower())
                    # Find the next closing tag after keyword
                    close_idx = html_body.find("</", idx + len(keyword))
                    if close_idx > 0:
                        tag_end = html_body.find(">", close_idx) + 1
                        img_tag = f'<br><img src="cid:{cid}" style="max-width:100%;height:auto;"><br>'
                        html_body = html_body[:tag_end] + img_tag + html_body[tag_end:]
                    break
    return html_body


def send_email(subject, html_body, summary):
    """Send email via Gmail SMTP with embedded chart images."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage

    mail_username = os.environ.get("MAIL_USERNAME")
    mail_password = os.environ.get("MAIL_PASSWORD")
    mail_port = int(os.environ.get("MAIL_PORT", "587"))
    email_to = os.environ.get("EMAIL_TO", "")

    if not mail_username or not mail_password:
        print("MAIL_USERNAME/MAIL_PASSWORD not set. Saving email to file instead.")
        save_email_to_file(subject, html_body)
        return

    if not email_to:
        print("EMAIL_TO not set. Saving email to file instead.")
        save_email_to_file(subject, html_body)
        return

    recipients = [addr.strip() for addr in email_to.split(",") if addr.strip()]

    # Build MIME message
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = mail_username
    msg["To"] = ", ".join(recipients)

    # Attach HTML body
    msg.attach(MIMEText(html_body, "html"))

    # Attach charts as inline images
    for i, chart_name in enumerate(summary.get("charts", [])):
        chart_path = OUTPUT_DIR / chart_name
        if chart_path.exists():
            with open(chart_path, "rb") as f:
                img = MIMEImage(f.read(), _subtype="png")
            img.add_header("Content-ID", f"<chart_{i}>")
            img.add_header("Content-Disposition", "inline", filename=chart_name)
            msg.attach(img)

    # Send via SMTP
    try:
        with smtplib.SMTP("smtp.gmail.com", mail_port) as server:
            server.starttls()
            server.login(mail_username, mail_password)
            server.send_message(msg, from_addr=mail_username, to_addrs=recipients)
        print(f"Email sent to {recipients}")
    except Exception as e:
        print(f"Email send failed: {e}")
        save_email_to_file(subject, html_body)


def save_email_to_file(subject, html_body):
    """Save email as HTML file when SendGrid is not available."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = OUTPUT_DIR / f"email_report_{date_str}.html"

    full_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{subject}</title></head>
<body>
<p style="color:#999;font-size:12px;">Subject: {subject}</p>
<hr>
{html_body}
</body>
</html>"""

    with open(path, "w") as f:
        f.write(full_html)
    print(f"Email saved to {path}")


def run():
    """Main pipeline."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set")
        sys.exit(1)

    print(f"=== Generating Email Report via Groq ({MODELO}) ===\n")

    summary = load_summary()
    print(f"Loaded summary: {summary['run_date']}, prob={summary['bic_probability']}%")

    result = generate_analysis(summary)
    subject = result["subject"]
    html_body = result["html_body"]

    # Embed chart references
    html_body = embed_charts_in_html(html_body, summary)

    print(f"\nSubject: {subject}")
    print(f"Body length: {len(html_body)} chars")

    # Send or save
    send_email(subject, html_body, summary)

    # Also save the raw output
    with open(OUTPUT_DIR / "email_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    run()

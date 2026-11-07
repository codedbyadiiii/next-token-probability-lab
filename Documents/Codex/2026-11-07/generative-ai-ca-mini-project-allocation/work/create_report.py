from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    PageBreak, Table, TableStyle, KeepTogether, HRFlowable)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas
from pathlib import Path

OUT = Path('/Users/adityanalge/Documents/Codex/2026-11-07/generative-ai-ca-mini-project-allocation/outputs/06_Akash_Kurane_Next-Token-Probability-Lab.pdf')
OUT.parent.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, PALE = HexColor('#15233D'), HexColor('#2E67B1'), HexColor('#EAF1F9')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('CoverTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=25, leading=31, alignment=TA_CENTER, textColor=colors.black, spaceAfter=12))
styles.add(ParagraphStyle('CoverSub', parent=styles['Normal'], fontSize=11, leading=16, alignment=TA_CENTER, spaceAfter=28))
styles.add(ParagraphStyle('Section', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=23, textColor=colors.black, spaceBefore=4, spaceAfter=10))
styles.add(ParagraphStyle('Sub', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle('Body', parent=styles['BodyText'], fontName='Helvetica', fontSize=9.4, leading=14, spaceAfter=7))
styles.add(ParagraphStyle('Small', parent=styles['BodyText'], fontSize=8.2, leading=11, spaceAfter=4))
styles.add(ParagraphStyle('CenterSmall', parent=styles['Small'], alignment=TA_CENTER))
styles.add(ParagraphStyle('Code', parent=styles['BodyText'], fontName='Courier', fontSize=7.5, leading=10, backColor=HexColor('#F4F6F8'), borderPadding=8, spaceAfter=8))

def P(text, style='Body'): return Paragraph(text, styles[style])
def bullets(items): return [P('&bull; ' + x, 'Body') for x in items]

def header_footer(canvas, doc):
    canvas.saveState(); w, h = A4
    canvas.setFont('Helvetica', 9); canvas.setFillColor(NAVY)
    canvas.drawCentredString(w/2, h-42, 'P6 - GENERATIVE AI MINI-PROJECT')
    canvas.setStrokeColor(HexColor('#C8D4E3')); canvas.line(54, h-50, w-54, h-50)
    canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.grey)
    canvas.drawCentredString(w/2, 33, f'Next-Token Probability Lab  |  Page {doc.page}')
    canvas.restoreState()

class Report(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, rightMargin=54, leftMargin=54, topMargin=70, bottomMargin=52, title='Next-Token Probability Lab Project Report', author='Akash Mahadeo Kurane')
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id='normal')
        self.addPageTemplates([PageTemplate(id='main', frames=frame, onPage=header_footer)])

def table(data, widths):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('TEXTCOLOR',(0,0),(-1,0),colors.black),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),8.5),('LEADING',(0,0),(-1,-1),11),('GRID',(0,0),(-1,-1),0.35,HexColor('#BAC9D8')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,HexColor('#FAFCFE')]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    return t

story=[]
# Cover
story += [Spacer(1, 18), P('P6 - GENERATIVE AI MINI-PROJECT', 'CenterSmall'), Spacer(1, 38), P('Next-Token Probability Lab', 'CoverTitle'), P('An interactive application for exploring token probability, self-attention, perplexity, sampling, model comparison, and safety in language models.', 'CoverSub'), Spacer(1, 22)]
info = [['Name','Akash Mahadeo Kurane'],['Class','LY AIA'],['Enrolment No.','ADT23SOCB0094'],['Roll No.','06'],['Subject','Generative AI']]
story.append(table(info, [1.55*inch, 4.95*inch])); story += [Spacer(1, 62), P('PROJECT REPORT - WITH DEPLOYMENT EVIDENCE', 'CenterSmall'), Spacer(1, 20)]
story += [P('Problem and Objective','Section'), P('Project title: Next-Token Probability Lab (P6). The project objective is to build an educational Streamlit application that makes the internal behavior of autoregressive and masked language models inspectable. The application presents next-token probabilities, final-layer attention, perplexity, BERT-versus-GPT-2 predictions, controlled PyTorch sampling, generated visual output, and safety analysis in one interface.', 'Body'), P('Project objectives','Sub')]
story += bullets(['Calculate and visualize the next-token distribution of GPT-2 under temperature and top-k controls.', 'Extract and display final-layer self-attention weights and calculate perplexity for an input prompt.', 'Compare BERT masked-token predictions with GPT-2 causal next-token predictions.', 'Implement text generation manually with PyTorch sampling and create a text-to-image result using Pillow.', 'Apply a lightweight safety filter and discuss how sampling temperature affects factual reliability.'])
story += [P('Scope and interpretation','Sub'), P('This application is an educational laboratory rather than a production moderation, search, or factuality system. Probability, attention, and perplexity values are model-specific. Generated text can be biased or factually incorrect and must be reviewed before use.', 'Body'), PageBreak()]

story += [P('1. System Overview','Section'), P('The application is organized into a presentation layer and an inference layer. Streamlit handles layout, controls, charts, warnings, and user interaction. The models module handles model loading, tokenization, inference, sampling, safety checks, and Pillow image generation.', 'Body')]
story.append(table([['Layer','Implementation','Responsibility'],['User interface','app.py / Streamlit','Controls, sections, charts, metrics, warnings, and responsive glassmorphism styling'],['Inference logic','models.py / PyTorch + Transformers','Model loading, probabilities, attention, perplexity, BERT predictions, generation, safety, and image rendering'],['Visualization','Plotly, Matplotlib, Seaborn, Pillow','Interactive probability chart, attention heatmap, and generated text canvas'],['Models','gpt2 and bert-base-uncased','Causal language modeling and masked language modeling']], [1.35*inch,1.75*inch,3.4*inch]))
story += [Spacer(1, 16), P('Processing flow','Sub'), P('Prompt -> GPT-2 tokenizer -> model logits -> temperature scaling -> top-k/top-p filtering -> probability chart or multinomial sampling -> safety filter -> displayed text and image.', 'Code'), P('Design principles','Sub')]
story += bullets(['Separation of concerns: UI code remains in app.py and inference code remains in models.py.', 'CPU-safe plotting: tensors are detached and moved to CPU before NumPy or visualization use.', 'Graceful failure: empty inputs, missing [MASK] tokens, and model-loading failures are handled with user-facing messages.', 'Minimal dependencies: the implementation uses only the libraries needed for the specified features.'])
story += [PageBreak(), P('2. U1 - Probability Lab and Sliders','Section'), P('The U1 section accepts a text prompt and evaluates the final GPT-2 logits at the last input position. The logits are divided by temperature before softmax. The top-k candidate token IDs are then decoded and displayed with their normalized probabilities.', 'Body'), P('Controls','Sub')]
story.append(table([['Control','Range','Effect'],['Temperature','0.1 to 2.0','Low values concentrate probability; high values flatten the distribution.'],['Top-k','1 to 100','Keeps only the k highest-scoring candidate tokens.'],['Top-p','0.1 to 1.0','Keeps the smallest candidate set whose cumulative probability reaches p.']], [1.35*inch,1.25*inch,3.9*inch]))
story += [Spacer(1, 12), P('Implementation evidence','Sub'), P('The horizontal Plotly chart ranks decoded candidate tokens and annotates each bar with a percentage. Tokens are displayed using repr-style labels so whitespace and punctuation remain visible to the learner.', 'Body'), P('Example interpretation','Sub'), P('If a prompt produces one token at 0.42 probability and several smaller alternatives, the model is relatively concentrated at that position. Increasing temperature makes alternative continuations more competitive; it does not create new knowledge.', 'Body'), PageBreak()]

story += [P('3. U2 - Attention and Perplexity','Section'), P('GPT-2 is called with output_attentions enabled. The attention tensor from the final transformer layer is averaged across heads to form a token-by-token matrix. This matrix is plotted as a Seaborn heatmap after moving it to CPU.', 'Body'), P('Perplexity calculation','Sub'), P('For a sequence, the model loss is the mean negative log-likelihood of the target tokens. Perplexity is computed as exp(loss). Lower values indicate that the model assigned higher likelihood to the observed sequence under the same tokenizer and evaluation conditions.', 'Body'), P('Comparison with a standard LSTM baseline','Sub'), P('An LSTM language model can report the same perplexity metric, but values should only be compared when the tokenization, corpus, sequence handling, and evaluation split are equivalent. GPT-2 uses self-attention and can model long-range relationships differently from a standard recurrent LSTM.', 'Body'), P('Interpretation caution','Sub'), P('Attention is a useful diagnostic visualization, but a bright cell should not automatically be interpreted as a complete causal explanation of the model prediction. It shows where a particular layer and head-averaged representation placed weight for the selected input.', 'Body'), PageBreak()]

story += [P('4. U3 - BERT versus GPT-2','Section'), P('This section demonstrates two different language-modeling objectives. BERT receives a sentence containing the literal [MASK] token and predicts the missing token using bidirectional context. GPT-2 receives the same sentence with the mask removed and predicts the next token using only the left-to-right context.', 'Body')]
story.append(table([['Model','Objective','Context available','Output shown'],['BERT','Masked language modeling','Tokens on both sides of [MASK]','Top five replacements for the mask'],['GPT-2','Causal language modeling','Only preceding tokens','Top five next-token candidates']], [1.0*inch,1.65*inch,2.2*inch,1.65*inch]))
story += [Spacer(1, 12), P('Example input','Sub'), P('The capital of France is [MASK].', 'Code'), P('The side-by-side display makes it clear that a masked model can use right-hand context, while a causal model is designed to continue a sequence from its prefix. The scores are model probabilities, not guaranteed truth values.', 'Body'), PageBreak()]

story += [P('5. U4 - PyTorch Sampling and Text-to-Image','Section'), P('The generation implementation does not use the Hugging Face pipeline. It manually performs the autoregressive loop: encode the prompt, run the model, select the last-position logits, apply temperature, apply top-k and top-p filtering, sample with torch.multinomial, append the next token, and stop at the end-of-sequence token or the requested token limit.', 'Body'), P('Generation algorithm','Sub')]
story += [P('1. Encode the prompt and move input IDs to the selected device.<br/>2. Run GPT-2 on the current context.<br/>3. Scale logits by temperature.<br/>4. Remove tokens outside top-k and the nucleus top-p set.<br/>5. Convert filtered logits to probabilities and sample with torch.multinomial.<br/>6. Append the sampled ID and repeat.', 'Body'), P('Text-to-image output','Sub'), P('The final generated text is drawn onto a dynamically created Pillow canvas with a dark gradient-like background, rounded panel, title, and wrapped text. The image is generated programmatically for each successful result and displayed in Streamlit.', 'Body'), PageBreak()]

story += [P('6. U5 - Safety Guardrails and Bias','Section'), P('Before generated text is displayed, it is passed through a lightweight keyword-based safety filter. The filter is intentionally conservative and checks for selected violent-threat and targeted-abuse patterns. When triggered, the application blocks the output and displays a warning rather than rendering the generated content.', 'Body'), P('Why a fallback filter?','Sub'), P('The project keeps third-party dependencies minimal. A production system would require a separately evaluated moderation model, multilingual coverage, adversarial testing, logging, human review paths, and clear governance. The current filter is an educational guardrail, not a complete safety solution.', 'Body'), P('Temperature and factual errors','Sub'), P('At low temperatures, generation favors high-probability continuations. This can improve consistency but may repeat a confident misconception. At high temperatures, the distribution is flatter and the output is more diverse, but unsupported claims, contradictions, and hallucinations become more likely. Temperature changes sampling behavior; it does not improve the model’s underlying factual knowledge.', 'Body'), P('Bias considerations','Sub'), P('GPT-2 and BERT learn statistical patterns from their training data. Their predictions can reflect stereotypes, uneven representation, and historical bias. Probability charts and attention plots should therefore be interpreted as model behavior, not as objective measures of truth or social value.', 'Body'), PageBreak()]

story += [P('7. User Interface and Visual Design','Section'), P('The Streamlit interface uses custom CSS to create an Apple-inspired presentation: clean typography, soft blue gradients, rounded glass-like cards, restrained borders, and responsive columns. The UI is divided into five task sections so that a learner can move from token probabilities to attention, model comparison, generation, and safety analysis.', 'Body'), P('Interaction design','Sub')]
story += bullets(['Sidebar controls remain visible while the main workspace presents the analysis.', 'Buttons are used for expensive operations such as BERT loading and text generation.', 'Charts and images use the full available width for readability.', 'Warnings are shown in context when input validation or safety filtering prevents an operation.'])
story += [P('Error handling','Sub'), P('The application checks for empty prompts, validates the presence of [MASK] for BERT, limits generation length, handles model-loading errors, and protects visualization code with user-facing exception messages. The models are cached as Streamlit resources so repeated interactions do not reload them.', 'Body'), PageBreak()]

story += [P('8. Testing and Deployment','Section'), P('The source files were checked for Python syntax validity. The application is designed to run locally or on Streamlit Community Cloud with no API keys.', 'Body'), P('Test checklist','Sub')]
story.append(table([['Test case','Expected result'],['Empty prompt','A validation message asks the user to enter a prompt.'],['BERT input without [MASK]','A warning explains that the literal [MASK] token is required.'],['Top-k and top-p controls','Values remain within the specified ranges and sampling completes.'],['Safety trigger','Generated output is blocked and a warning is displayed.'],['CPU-only machine','Models and plots work using CPU tensors; plotting receives CPU data.'],['First launch','Models download from Hugging Face and are cached for later use.']], [2.1*inch,4.0*inch]))
story += [Spacer(1, 12), P('Local deployment','Sub'), P('Install the dependencies with pip install -r requirements.txt and launch the interface with streamlit run app.py. The first run requires internet access to download gpt2 and bert-base-uncased. A deployment URL can be generated with Streamlit Community Cloud by selecting the GitHub repository, main branch, and app.py entry point.', 'Body'), PageBreak()]

story += [P('9. Limitations and Future Scope','Section'), P('Current limitations','Sub')]
story += bullets(['The safety filter is keyword-based and can miss context, coded language, or multilingual unsafe content.', 'GPT-2 and BERT are relatively small educational models and are not suitable for high-stakes factual systems.', 'Perplexity is meaningful only under matched tokenization and evaluation conditions.', 'Attention visualizations are diagnostic summaries and should not be treated as complete explanations.', 'Model downloads and CPU inference can take time on first launch.'])
story += [P('Future improvements','Sub')]
story += bullets(['Add an evaluated toxicity or safety classifier with documented thresholds.', 'Add support for multiple prompts, downloadable charts, and experiment history.', 'Add GPU-aware batching and session-level performance metrics.', 'Add a formal evaluation dataset for perplexity, factuality, and safety behavior.', 'Add authentication and rate limiting before public deployment.'])
story += [P('Conclusion','Sub'), P('Next-Token Probability Lab combines model introspection and controlled generation in a compact educational application. It demonstrates how GPT-2 probabilities, attention, perplexity, BERT masking, PyTorch sampling, and safety reasoning fit together in an end-to-end Generative AI workflow.', 'Body'), PageBreak()]

story += [P('References','Section'), P('1. Radford, A. et al. Improving Language Understanding by Generative Pre-Training. OpenAI, 2018.', 'Body'), P('2. Devlin, J. et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019.', 'Body'), P('3. Wolf, T. et al. Transformers: State-of-the-Art Natural Language Processing. EMNLP, 2020.', 'Body'), P('4. PyTorch Documentation. Tensors, inference mode, top-k selection, and multinomial sampling.', 'Body'), P('5. Streamlit Documentation. Application deployment and caching resources.', 'Body'), P('6. Hugging Face Model Cards. gpt2 and bert-base-uncased.', 'Body'), Spacer(1, 25), HRFlowable(width='100%', thickness=0.7, color=HexColor('#C8D4E3')), Spacer(1, 15), P('Project repository: https://github.com/codedbyadiiii/next-token-probability-lab', 'Small'), P('Subject: Generative AI  |  Roll No. 06  |  Project: P6', 'Small')]

Report(str(OUT)).build(story)
print(OUT)

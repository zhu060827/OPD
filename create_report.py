from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE
from pathlib import Path

out = Path('/Users/zhuxiyao/大创/结题文件代码/four_expert_evaluation_report.docx')
doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.75); sec.bottom_margin = Inches(0.7)
sec.left_margin = Inches(0.85); sec.right_margin = Inches(0.85)

styles = doc.styles
styles['Normal'].font.name = 'Arial Unicode MS'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), 'Arial Unicode MS'); styles['Normal'].font.size = Pt(10.5)
styles['Normal'].paragraph_format.space_after = Pt(6); styles['Normal'].paragraph_format.line_spacing = 1.15
for sname, size in [('Title',20),('Heading 1',14),('Heading 2',11.5)]:
    st=styles[sname]; st.font.name='Arial Unicode MS'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'Arial Unicode MS'); st.font.size=Pt(size); st.font.bold=True; st.font.color.rgb=RGBColor(0,0,0)
styles['Title'].paragraph_format.space_after=Pt(4)
# remove built-in title border/rule for a clean report title
title_pPr = styles['Title']._element.get_or_add_pPr()
for child in list(title_pPr):
    if child.tag == qn('w:pBdr'):
        title_pPr.remove(child)

# helpers
def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), fill); tcPr.append(shd)
def borders(table, color='D9D9D9'):
    tblPr=table._tbl.tblPr; b=tblPr.first_child_found_in('w:tblBorders')
    if b is None: b=OxmlElement('w:tblBorders'); tblPr.append(b)
    for edge in ('top','left','bottom','right','insideH','insideV'):
        tag='w:'+edge; el=b.find(qn(tag))
        if el is None: el=OxmlElement(tag); b.append(el)
        el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:space'),'0'); el.set(qn('w:color'),color)
def set_cell(cell, text, bold=False, color=None, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text=''; p=cell.paragraphs[0]; p.alignment=align; p.paragraph_format.space_after=Pt(2)
    r=p.add_run(str(text)); r.bold=bold; r.font.size=Pt(9)
    if color: r.font.color.rgb=RGBColor(*color)
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(headers, rows, widths=None):
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'; borders(t)
    for i,h in enumerate(headers): set_cell(t.rows[0].cells[i],h,True,(255,255,255)); shade(t.rows[0].cells[i],'244062')
    for ridx,row in enumerate(rows):
        cells=t.add_row().cells
        for i,v in enumerate(row): set_cell(cells[i],v,False,None,WD_ALIGN_PARAGRAPH.CENTER)
        if ridx%2==1:
            for c in cells: shade(c,'F2F6FA')
    if widths:
        for row in t.rows:
            for i,w in enumerate(widths): row.cells[i].width=Inches(w)
    doc.add_paragraph().paragraph_format.space_after=Pt(2)
    return t

p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('四个代码重写专家模型实验报告')
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run('基于 Qwen3-4B 与 LoRA 的领域专家评测'); r.italic=True; r.font.size=Pt(11); r.font.color.rgb=RGBColor(89,89,89)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('实验版本：Formal v1  |  评测对象：独立测试集').font.size=Pt(9)

doc.add_heading('1 研究目的与结论', level=1)
doc.add_paragraph('本报告评估四个面向代码重写任务的领域专家模型，分别覆盖格式修复、标识符恢复、局部结构变换和条件控制流改写。每个专家均与同一 Qwen3-4B 基座模型进行对照，在未参与训练的测试集上使用相同生成设置完成评测。')
doc.add_paragraph('结果表明，LoRA 专家在四个任务上的主指标均优于基座模型：Formatting、Identifier、Local Structure 和 Control Flow 的绝对提升分别为 50.34、27.06、15.37 和 33.90 个百分点。结构门控指标整体保持在 98.88% 以上，说明性能提升没有伴随大规模的代码结构失效。')

doc.add_heading('2 四个专家的职责分工', level=1)
doc.add_paragraph('四个专家共享同一个 Qwen3-4B 基座模型和 LoRA 训练框架，但分别针对不同类型的代码重写缺陷进行专门化训练。它们的职责如下：')
add_table(['专家','负责内容','输入问题','主要目标'],[
['Formatting','代码格式修复','缩进、空格、换行或代码风格存在问题','保持程序逻辑和 AST 结构不变，恢复目标格式'],
['Identifier','标识符恢复','变量名、函数名或参数名被替换、破坏或不规范','恢复语义合理且与目标一致的标识符'],
['Local Structure','局部结构变量提取','局部变量定义或使用关系需要补全、恢复','恢复局部变量结构及其依赖关系'],
['Control Flow','条件控制流改写','if、条件表达式及相关控制流需要重写','保持程序语义和测试行为的前提下改写控制流'],
],[1.0,1.35,2.0,2.1])
doc.add_paragraph('因此，Formatting 主要衡量格式保持能力，Identifier 主要衡量标识符恢复能力，Local Structure 主要衡量局部代码结构恢复能力，Control Flow 主要衡量条件改写后的可执行性和测试通过能力。')

doc.add_heading('3 数据与训练设置', level=1)
doc.add_paragraph('数据按 problem_id 进行划分，并对 train、validation 和 test 三个集合执行问题标识符与目标代码哈希重叠审计。四个专家的正式训练均采用 Qwen3-4B、4-bit QLoRA、LoRA rank 16、LoRA alpha 32、学习率 2×10⁻⁴、assistant-only causal language modeling loss。')
add_table(['专家','任务','Train','Validation','Test','最佳 checkpoint'],[
['Formatting','格式修复','5,892','736','737','checkpoint-900'],
['Identifier','标识符恢复','5,835','722','728','checkpoint-700'],
['Local Structure','局部结构变量提取','11,085','1,385','1,386','checkpoint-600'],
['Control Flow','条件控制流改写','4,393','567','534','checkpoint-500'],
],[1.0,1.6,.7,.8,.6,1.2])
doc.add_heading('3.1 专家训练数据集记录', level=2)
doc.add_paragraph('四个专家分别使用以下正式训练集和验证集。这里的“来源”指原始代码数据集及其领域专用构造方式，不仅是服务器上的文件路径。数据按 problem_id 划分，并在训练前完成 problem_id 与目标代码哈希重叠审计。测试集仅用于最终 Base/LoRA 对照评测，不参与训练或校准。')
add_table(['专家','原始数据来源','领域构造方式','训练范式依据'],[
['Formatting','BigCode CommitPackFT（Python）','真实格式违规版本或确定性格式扰动 → 规范格式；保持标识符、AST 语义和控制流不变','STYLER；Learning Natural Coding Conventions'],
['Identifier','CodeSearchNet（Python）','DOBF 式局部变量/参数混淆 → 原始自然标识符；记录作用域和定义-使用位置','DOBF'],
['Local Structure','CodeSearchNet（Python）','可逆 Extract/Inline、临时变量引入/消除和局部语句变换','NatGen'],
['Control Flow','DeepMind CodeContests（Python 正确解答）','guard clause、conditional rewrite、loop、early return、break/continue 等语义保持变换','NatGen；ContraCode'],
],[1.0,1.7,3.0,1.45])
doc.add_paragraph('来源登记：Formatting 使用 bigcode/commitpackft；Identifier 和 Local Structure 使用 code-search-net/code_search_net；Control Flow 使用 deepmind/code_contests。原始数据集仅提供代码及其原生信息，训练样本由项目内的专用适配器和确定性构造器生成，并通过解析、编译、接口一致性及领域特定语义门禁。')
add_table(['专家','训练集文件','验证集文件','训练样本','验证样本'],[
['Formatting','teacher_training/data/processed/formatting/train.jsonl','teacher_training/data/processed/formatting/validation.jsonl','5,892','736'],
['Identifier','teacher_training/data/processed/identifier/train.jsonl','teacher_training/data/processed/identifier/validation.jsonl','5,835','722'],
['Local Structure','teacher_training/data/processed/local_structure/train.jsonl','teacher_training/data/processed/local_structure/validation.jsonl','11,085','1,385'],
['Control Flow','teacher_training/data/processed/control_flow/train.jsonl','teacher_training/data/processed/control_flow/validation.jsonl','4,393','567'],
],[1.05,2.55,2.65,.75,.8])
doc.add_paragraph('服务器项目根目录：/home/tangyafeng/zhuxiyao/teacher-specialization；上述文件均位于该目录下。对应的正式测试集分别为 formatting/test.jsonl、identifier/test.jsonl、local_structure/test.jsonl 和 control_flow/test.jsonl。')

doc.add_heading('4 主指标结果', level=1)
doc.add_paragraph('主指标根据任务性质定义：Formatting 使用精确格式修复率，Identifier 使用精确标识符恢复率，Local Structure 使用目标代码 Exact Match，Control Flow 使用 Pass@1/Test Pass。')
add_table(['专家','主指标','Base','LoRA','提升'],[
['Formatting','Exact Formatting Repair','38.26%','88.60%','+50.34 pp'],
['Identifier','Exact Identifier Recovery','24.73%','51.79%','+27.06 pp'],
['Local Structure','Exact Match','84.42%','99.78%','+15.37 pp'],
['Control Flow','Pass@1 / Test Pass','63.48%','97.38%','+33.90 pp'],
],[1.1,1.9,0.8,0.8,0.9])

doc.add_heading('5 辅助指标与结构门控', level=1)
add_table(['专家','Compile Base','Compile LoRA','Parse Base','Parse LoRA','Signature Base','Signature LoRA'],[
['Formatting','99.46%','99.86%','99.46%','99.86%','99.32%','99.86%'],
['Identifier','99.31%','100.00%','99.31%','100.00%','97.66%','100.00%'],
['Local Structure','99.49%','100.00%','99.49%','100.00%','99.49%','100.00%'],
['Control Flow','99.06%','98.88%','99.06%','98.88%','98.88%','98.88%'],
],[1.15,.95,.95,.85,.85,1.05,1.05])
doc.add_paragraph('Formatting 的平均 Levenshtein Distance 从 21.23 降至 2.74；Identifier 的 Subtoken F1 从 0.3681 提升至 0.6034；Local Structure 的平均 CodeBLEU 从 0.9847 提升至 0.9998；Control Flow 的 CodeBLEU 从 0.7301 提升至 0.7737。')

doc.add_heading('6 结果分析', level=1)
doc.add_heading('5.1 Formatting', level=2); doc.add_paragraph('Formatting 专家将精确格式修复率从 38.26% 提升至 88.60%，同时 Compile、Parse 和 Signature Pass 均达到约 99.86%。编辑距离的大幅下降表明模型不仅生成了可解析代码，也更接近目标格式。')
doc.add_heading('5.2 Identifier', level=2); doc.add_paragraph('Identifier 专家将精确恢复率从 24.73% 提升至 51.79%，Subtoken F1 从 0.3681 提升至 0.6034。Compile、Parse 和 Signature Pass 均达到 100%，说明标识符恢复并未破坏代码接口和语法结构。')
doc.add_heading('5.3 Local Structure', level=2); doc.add_paragraph('Local Structure 的主指标从 84.42% 提升至 99.78%。由于基座模型已经处于较高水平，该任务存在明显的天花板效应；从错误率角度看，失败比例由 15.58% 降至 0.22%，实际错误减少约 98.6%。')
doc.add_heading('5.4 Control Flow', level=2); doc.add_paragraph('Control Flow 的 Pass@1/Test Pass 从 63.48% 提升至 97.38%，提升 33.90 个百分点。LoRA 的 Compile、Parse 和 Signature Pass 为 98.88%，相较基座模型低约 0.19 个百分点，说明少量复杂样本仍存在结构回归。')

doc.add_heading('7 失败案例与局限性', level=1)
doc.add_paragraph('Control Flow 失败样本主要包括两类：一是生成达到 2048 token 上限后被截断，导致代码标签不完整或程序不完整；二是少量局部语法错误，例如括号缺失。上述样本应保留在误差分析中，而不应从测试结果中删除。')
doc.add_paragraph('当前实验的主要局限包括：生成长度限制可能影响复杂程序；不同任务的数据规模不完全相同；本报告给出的是独立测试集上的描述性结果，尚未加入 paired bootstrap 或 paired sign test 的置信区间。')

doc.add_heading('8 实验产物与复现记录', level=1)
doc.add_paragraph('每个专家均保留了预测文件、Base/LoRA 的 summary.json、per_sample.csv、训练配置、最佳 checkpoint、训练状态和评测图表。统一汇总文件包括 experts_summary.csv、experts_summary.json、experiment_manifest.csv 和 experiment_manifest.json。')

doc.add_heading('9 总结', level=1)
doc.add_paragraph('四个领域专家均显著提升了对应代码重写任务的主指标。Formatting 和 Control Flow 获得最大绝对提升，Identifier 在精确恢复率与 Subtoken F1 上均有明显改善，Local Structure 则接近饱和性能。整体结果支持使用领域专用 LoRA 专家增强代码重写能力，同时表明生成长度和复杂控制流仍是后续优化重点。')

# footer
for section in doc.sections:
    footer=section.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=footer.add_run('四个代码重写专家模型实验报告'); run.font.size=Pt(8); run.font.color.rgb=RGBColor(128,128,128)

doc.core_properties.title='四个代码重写专家模型实验报告'
doc.core_properties.subject='Qwen3-4B LoRA code rewriting experts evaluation'
doc.core_properties.author='Teacher Specialization Project'
doc.save(out)
print(out)

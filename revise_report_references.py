from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import re
from pathlib import Path

src=Path('/Users/zhuxiyao/大创/附件6：哈尔滨工业大学大学生创新训练计划项目验收书.docx')
out=Path('/Users/zhuxiyao/大创/结题文件代码/附件6_结题报告_参考文献与题注修订版.docx')
doc=Document(src)

# Old citation number -> new citation number. Unlisted old numbers are removed.
number_map={1:1,14:2,28:3,30:4,15:5,22:6,7:7,23:8,27:9,34:10,32:11,33:12,35:13,11:14,36:15,37:16,38:17,39:18,40:19,41:20,42:21,43:22}

special_rewrites={
'传统数据增强依赖人工预设规则': '传统数据增强依赖人工预设规则，在视觉和语言等领域已得到广泛应用，但表格、数学推理和代码数据具有更严格的质量约束：表格特征需具有预测价值[1,2]，数学推理需保证过程与答案一致[3]，代码则需满足语法、接口和执行要求[4]。缺乏质量控制的数量扩充可能引入无效特征、错误推理或不可执行代码。',
'大语言模型能够生成多样化候选内容': '大语言模型能够生成多样化候选内容，但其输出存在不确定性[5,6]。因此，如何结合任务约束、执行结果和下游性能建立生成、验证与反馈闭环，是提高异构任务数据扩充质量的关键问题。',
'表格数据扩充：如图 2.2-1': '表格数据扩充：如图2.2-1，Nam等[7]利用决策树反馈引导大语言模型迭代优化特征规则；如图2.2-2，Hollmann等[5]提出上下文感知特征工程，结合数据集描述和任务背景生成具有语义解释的特征及程序；如图2.2-3，Huang等[8]利用大语言模型完成数据生成与质量优化，并通过检索增强生成提高事实准确性。现有方法仍面临候选空间大、计算成本高、生成结果不稳定和有效特征评价困难等问题。',
'数学推理扩充：思维链技术': '数学推理扩充：思维链技术通过生成中间步骤增强模型的复杂推理能力[9]。如图2.2-4，MathFimer采用中间填充方式在已有推理链的前后文之间恢复或补充中间步骤[10]，但现有方法主要基于线性步骤序列，对步骤间结构关系的表达有限，生成内容也可能出现逻辑错误、答案偏移和有效增量不足等问题。',
'代码数据扩充：HumanEval等工作': '代码数据扩充：HumanEval利用单元测试和实际执行结果评价代码的功能正确性[4]；知识蒸馏通过教师模型的输出分布指导学生模型训练[11]，在线策略蒸馏进一步在学生模型自身生成的轨迹上引入教师监督[12,13]。然而，代码扩充仍受到语法、接口、功能和安全等多重约束，文本相似性或单一质量指标难以充分反映程序的实际有效性。'
}

def replace_para_text(p,text):
    # Preserve paragraph properties/style while normalizing the text runs.
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run=p.add_run(text)
    run.font.name='宋体'
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    run.font.size=Pt(12)

# Reference heading boundary.
ref_idx=next(i for i,p in enumerate(doc.paragraphs) if p.text.strip()=='参考文献')

# Apply exact prose fixes first and exclude them from the legacy-number remap.
special_ids=set()
for p in doc.paragraphs[:ref_idx]:
    s=p.text.strip()
    for start,new in special_rewrites.items():
        if s.startswith(start):
            replace_para_text(p,new)
            special_ids.add(id(p._element))
            break

# Renumber remaining in-text citations.
def remap_group(m):
    content=m.group(1)
    nums=[]
    for part in re.split(r'[,，]\s*',content):
        part=part.strip()
        rr=re.fullmatch(r'(\d+)\s*[-–—]\s*(\d+)',part)
        if rr:
            nums.extend(range(int(rr.group(1)),int(rr.group(2))+1))
        elif part.isdigit():
            nums.append(int(part))
        else:
            return m.group(0)
    mapped=[]
    for n in nums:
        if n in number_map and number_map[n] not in mapped:
            mapped.append(number_map[n])
    if not mapped:
        return ''
    # Compact only truly consecutive runs of 3+; otherwise comma-separated.
    parts=[]; i=0
    while i<len(mapped):
        j=i
        while j+1<len(mapped) and mapped[j+1]==mapped[j]+1:
            j+=1
        if j-i>=2: parts.append(f'{mapped[i]}–{mapped[j]}')
        else: parts.extend(str(mapped[k]) for k in range(i,j+1))
        i=j+1
    return '['+','.join(parts)+']'

for p in doc.paragraphs[:ref_idx]:
    if id(p._element) in special_ids:
        continue
    old=p.text
    new=re.sub(r'\[([0-9,，\-–—\s]+)\]',remap_group,old)
    if new!=old:
        replace_para_text(p,new)

# Normalize figure/table caption numbering and separators throughout body.
caption_map={
    '图2.1-1：表格数据扩充任务的主要类型[1]':'图2.1-1 表格数据扩充任务的主要类型[1]',
    '图2.2-1: OCTree特征规则迭代优化框架[7]':'图2.2-1 OCTree特征规则迭代优化框架[7]',
    '图2.2-2：CAAFE自动特征工程框架[5]':'图2.2-2 CAAFE自动特征工程框架[5]',
    '图2.2-3: DataGen数据生成与质量优化框架[8]':'图2.2-3 DataGen数据生成与质量优化框架[8]',
    '图2.2-4: MathFimer数学推理步骤扩充框架[10]':'图2.2-4 MathFimer数学推理步骤扩充框架[10]',
    '图 2.2-5：EasyOPD统一在线策略蒸馏框架[13]':'图2.2-5 EasyOPD统一在线策略蒸馏框架[13]',
    '图 3.1-1：项目总技术研究框架':'图3.1-1 项目总体技术框架',
    '图 3.2-1：表格数据特征迭代扩充流程图':'图3.2-1 表格特征迭代扩充流程',
    '图3.3-1: 数学推理链迭代扩充流程图':'图3.3-1 数学推理链迭代扩充流程',
    '图3.4-1: 代码数据迭代扩充流程图':'图3.4-1 代码数据迭代扩充流程',
}
# Generic code-section renumbering and missing math figure number.
for p in doc.paragraphs[:ref_idx]:
    s=p.text.strip()
    if s in caption_map:
        replace_para_text(p,caption_map[s]); s=p.text.strip()
    new=s
    new=re.sub(r'^(图|表)\s*4\.3\.(\d+)\s*[:：]?\s*',lambda m:f'{m.group(1)}4.3-{m.group(2)} ',new)
    if new.startswith('图4.2-4 '):
        new='图4.2-3 '+new[len('图4.2-4 '):]
    if new.startswith('图4.3-4 '):
        new='图4.3-4 不同OPD训练配置的消融结果'
    if new.startswith('图4.3-5 '):
        new='图4.3-5 OPD训练过程中师生Top-k重合率与任务通过率'
    # Other captions: normalize colon to a space.
    if re.match(r'^(图|表)\d',new):
        new=re.sub(r'^((?:图|表)\d+(?:\.\d+)?-\d+)\s*[:：]\s*',r'\1 ',new)
    if new!=s:
        replace_para_text(p,new)
    if re.match(r'^(图|表)\d',p.text.strip()):
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before=Pt(3)
        p.paragraph_format.space_after=Pt(6)

# Normalize body cross-references to updated caption numbers.
for p in doc.paragraphs[:ref_idx]:
    old=p.text
    new=old
    for kind in ('图','表'):
        new=re.sub(fr'{kind}\s*4\.3\.(\d+)',fr'{kind}4.3-\1',new)
    new=new.replace('图4.2-4','图4.2-3')
    if new!=old:
        replace_para_text(p,new)

refs=[
'[1] CUI L, LI H, CHEN K, et al. Tabular Data Augmentation for Machine Learning: Progress and Prospects of Embracing Generative AI[EB/OL]. arXiv:2407.21523, 2024.',
'[2] ZHANG T, FAN Z, CHEN J, et al. OpenFE: Automated Feature Generation with Expert-level Performance[EB/OL]. arXiv:2211.12507, 2023.',
'[3] LIGHTMAN H, KOSARAJU V, BURDA Y, et al. Let’s Verify Step by Step[EB/OL]. arXiv:2305.20050, 2023.',
'[4] CHEN M, TWOREK J, JUN H, et al. Evaluating Large Language Models Trained on Code[EB/OL]. arXiv:2107.03374, 2021.',
'[5] HOLLMANN N, MÜLLER S, HUTTER F. Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering[C]//Advances in Neural Information Processing Systems. 2023, 36.',
'[6] REN J, ZHAO Y, VU T, et al. Self-Evaluation Improves Selective Generation in Large Language Models[EB/OL]. arXiv:2312.09300, 2023.',
'[7] NAM J, KIM K, OH S, et al. Optimized Feature Generation for Tabular Data via LLMs with Decision Tree Reasoning[C]//Advances in Neural Information Processing Systems. 2024, 37.',
'[8] HUANG Y, WU S, GAO C, et al. DataGen: Unified Synthetic Dataset Generation via Large Language Models[C]//International Conference on Learning Representations. 2025.',
'[9] WEI J, WANG X, SCHUURMANS D, et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models[C]//Advances in Neural Information Processing Systems. 2022, 35: 24824-24837.',
'[10] YAN Y, SHEN Y, LIU Y, et al. MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task[C]//International Conference on Learning Representations. 2026.',
'[11] HINTON G, VINYALS O, DEAN J. Distilling the Knowledge in a Neural Network[EB/OL]. arXiv:1503.02531, 2015.',
'[12] AGARWAL R, VIEILLARD N, ZHOU Y, et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes[C]//International Conference on Learning Representations. 2024.',
'[13] SUN J, ZHENG M, SONG M, et al. EasyOPD: An Easy-to-use On-Policy Distillation Framework for Large Language Models[EB/OL]. arXiv:2607.11012, 2026.',
'[14] CHEN T, GUESTRIN C. XGBoost: A Scalable Tree Boosting System[C]//Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York: ACM, 2016: 785-794.',
'[15] GRANDINI M, BAGLI E, VISANI G. Metrics for Multi-Class Classification: An Overview[EB/OL]. arXiv:2008.05756, 2020.',
'[16] BRODERSEN K H, ONG C S, STEPHAN K E, et al. The Balanced Accuracy and Its Posterior Distribution[C]//Proceedings of the 20th International Conference on Pattern Recognition. IEEE, 2010: 3121-3124.',
'[17] FAWCETT T. An Introduction to ROC Analysis[J]. Pattern Recognition Letters, 2006, 27(8): 861-874.',
'[18] GNEITING T, RAFTERY A E. Strictly Proper Scoring Rules, Prediction, and Estimation[J]. Journal of the American Statistical Association, 2007, 102(477): 359-378.',
'[19] AUSTIN J, ODENA A, NYE M, et al. Program Synthesis with Large Language Models[EB/OL]. arXiv:2108.07732, 2021.',
'[20] REN S, GUO D, LU S, et al. CodeBLEU: A Method for Automatic Evaluation of Code Synthesis[EB/OL]. arXiv:2009.10297, 2020.',
'[21] MCCABE T J. A Complexity Measure[J]. IEEE Transactions on Software Engineering, 1976, SE-2(4): 308-320.',
'[22] OMAN P, HAGEMEISTER J. Metrics for Assessing a Software System’s Maintainability[C]//Proceedings of the Conference on Software Maintenance. Orlando: IEEE, 1992: 337-344.'
]

# Refresh paragraph list after body changes.
pars=doc.paragraphs
ref_idx=next(i for i,p in enumerate(pars) if p.text.strip()=='参考文献')
old_ref_pars=pars[ref_idx+1:ref_idx+1+43]
assert len(old_ref_pars)==43
for i,ref in enumerate(refs):
    p=old_ref_pars[i]
    replace_para_text(p,ref)
    p.style=doc.styles['Normal']
    p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent=Cm(0.74)
    p.paragraph_format.first_line_indent=Cm(-0.74)
    p.paragraph_format.space_after=Pt(3)
    p.paragraph_format.line_spacing=1.0
    for run in p.runs:
        run.font.name='Times New Roman'
        run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
        run.font.size=Pt(10.5)

# Reuse next old reference paragraph as the separated appendix heading.
annex=old_ref_pars[len(refs)]
replace_para_text(annex,'六、附件（专利、发表论文及其他成果支撑材料）')
annex.style=doc.styles['Heading 1']
annex.alignment=WD_ALIGN_PARAGRAPH.LEFT
annex.paragraph_format.left_indent=Cm(0)
annex.paragraph_format.first_line_indent=Cm(0)
annex.paragraph_format.space_before=Pt(12)
annex.paragraph_format.space_after=Pt(6)

# Delete remaining obsolete reference paragraphs.
for p in old_ref_pars[len(refs)+1:]:
    el=p._element
    el.getparent().remove(el)

# Save.
doc.save(out)
print(out)

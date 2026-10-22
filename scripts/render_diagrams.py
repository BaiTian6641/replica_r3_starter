#!/usr/bin/env python3
"""Render the exact R3 architecture atlas as vector PDF and standalone SVGs.

Dependencies: reportlab. Optional pdftoppm produces PNG previews. The drawings
describe proposed architecture; executable primitive tests do not establish
pretrained-model quality. Re-run from any working directory.
"""
from __future__ import annotations

import html
import math
import os
from pathlib import Path
import shutil
import subprocess

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagrams"
W, H = 1200, 800
font_roots = [Path("/usr/share/fonts/truetype/liberation2"),
              Path("/usr/share/fonts/truetype/liberation"),
              Path(os.environ.get("CODEX_PRIMARY_RUNTIME_NODE_MODULES", "/nonexistent")) / "pdfjs-dist/standard_fonts",
              Path("/opt/codex/runtimes/codex-primary-runtime/dependencies/node/node_modules/pdfjs-dist/standard_fonts")]
font_root = next((p for p in font_roots if (p / "LiberationSans-Regular.ttf").is_file()), None)
if font_root:
    regular, bold = font_root / "LiberationSans-Regular.ttf", font_root / "LiberationSans-Bold.ttf"
else:
    font_root = Path(reportlab.__file__).parent / "fonts"
    regular, bold = font_root / "Vera.ttf", font_root / "VeraBd.ttf"
pdfmetrics.registerFont(TTFont("AtlasSans", str(regular)))
pdfmetrics.registerFont(TTFont("AtlasSans-Bold", str(bold)))
PAL = {
    "bg": "#F7F9FC", "ink": "#162A43", "muted": "#546B83", "line": "#69839C",
    "shipped": ("#E6F4EE", "#21856C"),
    "proposed": ("#EAF0FA", "#416EAB"),
    "planned": ("#FFF3D9", "#B88623"),
    "frozen": ("#EFF2F6", "#7E8DA0"),
    "barrier": ("#FCEEF0", "#B35665"),
}


class Drawing:
    def __init__(self, pdf, index, title, subtitle, slug):
        self.pdf, self.svg, self.slug = pdf, [], slug
        self.rect(0, 0, W, H, PAL["bg"], PAL["bg"], 0)
        self.rect(48, 38, 6, 72, "#21856C", "#21856C", 0)
        self.text(72, 47, "REPLICA  /  R3 RESEARCH ATLAS", 12, "#21856C", bold=True)
        self.text(72, 76, title, 30, PAL["ink"], bold=True)
        self.text(72, 116, subtitle, 14, PAL["muted"])
        self.text(1128, 48, f"{index:02d} / 08", 14, PAL["muted"], anchor="end")
        self.line([(48, 139), (1152, 139)], "#D5DFEA", 1)
        self._footer()

    def rect(self, x, y, w, h, fill, stroke, radius=12, width=1.5):
        self.pdf.setFillColor(HexColor(fill))
        self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.setLineWidth(width)
        if radius:
            self.pdf.roundRect(x, H-y-h, w, h, radius, fill=1, stroke=1)
        else:
            self.pdf.rect(x, H-y-h, w, h, fill=1, stroke=1)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def text(self, x, y, value, size=15, color=None, bold=False, anchor="start"):
        color = color or PAL["ink"]
        font = "AtlasSans-Bold" if bold else "AtlasSans"
        self.pdf.setFont(font, size)
        self.pdf.setFillColor(HexColor(color))
        if anchor == "end":
            self.pdf.drawRightString(x, H-y-size, value)
        elif anchor == "middle":
            self.pdf.drawCentredString(x, H-y-size, value)
        else:
            self.pdf.drawString(x, H-y-size, value)
        self.svg.append(f'<text x="{x}" y="{y+size}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{html.escape(value)}</text>')

    def line(self, points, color=None, width=2, dashed=False, arrow=False):
        color = color or PAL["line"]
        self.pdf.setStrokeColor(HexColor(color)); self.pdf.setFillColor(HexColor(color))
        self.pdf.setLineWidth(width)
        self.pdf.setDash(6, 4) if dashed else self.pdf.setDash()
        p = self.pdf.beginPath(); p.moveTo(points[0][0], H-points[0][1])
        for x, y in points[1:]: p.lineTo(x, H-y)
        self.pdf.drawPath(p)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash}/>')
        self.pdf.setDash()
        if arrow:
            (ax, ay), (bx, by) = points[-2:]
            angle = math.atan2(by-ay, bx-ax)
            corners = [(bx,by), (bx-10*math.cos(angle)+4*math.sin(angle),by-10*math.sin(angle)-4*math.cos(angle)),
                       (bx-10*math.cos(angle)-4*math.sin(angle),by-10*math.sin(angle)+4*math.cos(angle))]
            p = self.pdf.beginPath(); p.moveTo(corners[0][0],H-corners[0][1])
            for x,y in corners[1:]: p.lineTo(x,H-y)
            p.close(); self.pdf.drawPath(p,fill=1,stroke=0)
            self.svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in corners)}" fill="{color}"/>')

    def arrow(self, points, **kwargs): self.line(points, arrow=True, **kwargs)

    def box(self, x,y,w,h,title,body=(),kind="proposed",title_size=17,body_size=14):
        fill,stroke=PAL[kind]
        self.rect(x,y,w,h,fill,stroke)
        if isinstance(body,str): body=body.split("\n")
        titles = wrap(title,w-34,title_size,bold=True)
        lines=[]
        for part in body: lines.extend(wrap(part,w-34,body_size))
        need=15+len(titles)*(title_size+4)+7+len(lines)*(body_size+5)+10
        if need>h:
            raise ValueError(f"Overflow: {title!r} requires {need}px, box {h}px")
        yy=y+15
        for line in titles:
            self.text(x+17,yy,line,title_size,stroke,bold=True); yy+=title_size+4
        yy+=7
        for line in lines:
            self.text(x+17,yy,line,body_size); yy+=body_size+5

    def note(self,x,y,w,text,color=None,size=13):
        for i,line in enumerate(wrap(text,w,size)):
            self.text(x,y+i*(size+5),line,size,color or PAL["muted"])

    def _footer(self):
        self.line([(48,704),(1152,704)],"#D5DFEA",1)
        labels=[("shipped","Shipped primitives / CPU checks",72),
                ("proposed","Proposed system integration",445),
                ("planned","Optional research branch",815)]
        for kind,label,x in labels:
            fill,stroke=PAL[kind]
            self.rect(x,722,13,13,fill,stroke,3,1)
            self.text(x+23,720,label,12,PAL["muted"])
        self.text(72,752,"Proposed research architecture; CPU contract tests are not model-quality evidence.",12,PAL["muted"])
        self.text(1128,752,"23 OCT 2026",11,PAL["muted"],anchor="end")

    def finish(self):
        doc=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(self.slug)}">'+"\n".join(self.svg)+"</svg>\n"
        (OUT/f"{self.slug}.svg").write_text(doc,encoding="utf-8")
        self.pdf.showPage()


def wrap(value,width,size,bold=False):
    words=value.split(); lines=[]; current=""
    font="AtlasSans-Bold" if bold else "AtlasSans"
    for word in words:
        trial=(current+" "+word).strip()
        if stringWidth(trial,font,size)<=width: current=trial
        else:
            if current: lines.append(current)
            current=word
    if current: lines.append(current)
    return lines


def system(pdf):
    d=Drawing(pdf,1,"The complete experience-to-learning system",
              "Stable foundation, evidence-backed memory, a bounded control loop and separately owned identity.","01_full_system")
    d.box(60,180,260,105,"Goal + observations",["User request, current task","Permitted sensory / tool input"])
    d.box(440,180,320,105,"Controller + task workspace",["Goal, uncertainty, action choices","Ask, act, verify or stop"])
    d.box(880,180,260,105,"Backbone + readers",["Native text / visual inference","Read eligible experience"])
    d.box(880,365,260,105,"Tools + evidence",["Bounded external actions","Independent observations"])
    d.box(880,550,260,105,"Verified outcome",["Result, failure, uncertainty","Preserve source and event time"])
    d.box(440,550,320,105,"Episode / evidence store",["Temporal records and provenance","Durable state with owner + scope"],"shipped")
    d.box(60,550,260,105,"Learning director",["Choose a useful practice target","Budgeted candidate proposal"])
    d.box(440,365,320,115,"Candidate + evaluation",["Train only allowed parameter groups","Compare benefit, retention and fidelity","Promote, reject or roll back"])
    d.box(60,355,260,125,"Identity evidence",["Human source archive","Approved identity snapshot","Separate replica branch history"])
    for pts in [[(320,225),(440,225)],[(760,225),(880,225)],[(1010,285),(1010,365)],
                [(1010,470),(1010,550)],[(880,605),(760,605)],[(440,605),(320,605)],
                [(190,550),(190,515),(600,515),(600,480)],[(600,365),(600,285)],
                [(320,415),(390,415),(390,260),(440,260)]]:d.arrow(pts)
    d.arrow([(760,590),(815,590),(815,315),(700,315),(700,285)],color="#21856C")
    d.text(782,490,"read",12,"#21856C")
    d.text(620,325,"candidate revision",12)
    d.note(60,668,1080,"Source-person identity is evidence, not a self-generated biography. New agent experience belongs to the replica branch.")
    d.finish()


def internal(pdf):
    d=Drawing(pdf,2,"Small-model integration inside the native backbone",
              "Shipped text scaffold: one native Llama block receives two independently gated residual additions.","02_native_llama_integration")
    d.box(60,175,260,105,"Native hidden states",["Original embeddings and layers","Normal causal attention mask"],"frozen")
    d.box(420,175,300,105,"Native decoder block",["Preserve original computation","Base frozen or fully trainable"],"frozen")
    d.box(880,175,260,125,"Add independent deltas",["Native output + memory + workspace","Separate gates; zero-init outputs","Continue through native model"],"shipped",16,13)
    d.box(60,365,260,105,"Source-only embeddings",["Past memory records only","No future target tokens as input"],"shipped",16,13)
    d.box(420,365,300,105,"Record writer + slots",["Trainable projection / pooling","Fixed count of memory slots"],"shipped")
    d.box(880,365,260,105,"Memory reader",["Q = native block output","Read source-derived slots"],"shipped")
    d.box(60,550,260,105,"Prompt-only embeddings",["Observed prefix before targets","Supervised target positions rejected"],"shipped",16,13)
    d.box(420,550,300,105,"Bounded workspace",["Prompt-conditioned latent slots","Fixed recurrent refinement budget"],"shipped")
    d.box(880,550,260,105,"Workspace reader",["Q = native block output","Read task-local workspace"],"shipped")
    for y in (225,415,600):
        d.arrow([(320,y),(420,y)]); d.arrow([(720,y),(880,y)])
    d.arrow([(1140,415),(1162,415),(1162,245),(1140,245)],color="#21856C")
    d.arrow([(1140,600),(1182,600),(1182,275),(1140,275)],color="#21856C")
    d.note(70,315,1010,"Zero output initialization preserves native logits at initialization. It does not prove the added branches will improve quality.",size=13)
    d.note(70,670,1060,"One in-flight batch owns the context until backward completes. Save the native base and sidecars separately; cached generation is pending.",size=12)
    d.finish()


def timescales(pdf):
    d=Drawing(pdf,3,"Different states learn on different timescales",
              "Persistence, update authority and neural parameters are explicit rather than hidden inside one memory object.","03_memory_timescales")
    rows=[
        ("One task","Task workspace","Prompt-conditioned goal / context slots","Reset at task boundary; no durable write","shipped"),
        ("One chunk","Private fast tensors","Optional functional associative state","All reads use chunk start; then update","planned"),
        ("Across sessions","Evidence + episodes","Source text, time, lineage and owner","Append / supersede / forget by policy","shipped"),
        ("Between candidates","Skill / sidecar parameters","Reader, writer, policy or adapter weights","Sandbox training + independent promotion","proposed"),
        ("Versioned release","Backbone parameters","Pretrained or developmentally adapted base","Frozen in deployment; explicit new release","frozen"),
    ]
    for i,(time,title,body,rule,kind) in enumerate(rows):
        y=167+i*101
        d.text(65,y+18,time,16,PAL["muted"],bold=True)
        d.box(270,y,370,84,title,[body],kind,17,13)
        d.box(745,y,395,84,"Update contract",[rule],"frozen",16,13)
        d.arrow([(640,y+42),(745,y+42)])
    d.note(65,678,1060,"A stored fact, an online tensor update and a promoted parameter revision are different improvement claims.",size=13)
    d.finish()


def causal(pdf):
    d=Drawing(pdf,4,"Causal memory: support before query, commit after read",
              "A shared immutable chunk-start state prevents one query token from seeing a later write in the same chunk.","04_causal_support_query")
    d.line([(420,158),(420,651)],"#B35665",2,dashed=True)
    d.text(438,164,"CAUSAL BARRIER",12,"#B35665",bold=True)
    d.box(60,195,300,110,"Earlier support events",["Known before the query","Eligible observed records only"],"shipped")
    d.box(500,195,630,110,"Current query chunk",["Tokens are processed under the native causal mask","Every query read uses the same M[k] snapshot"],"planned")
    d.box(60,365,300,110,"Support writer",["No sealed answers or future events","No persistent mutation in forward"],"shipped")
    d.box(500,365,270,110,"M[k]: chunk-start state",["Immutable for all chunk reads","Private task / stream ownership"],"planned",16,13)
    d.box(880,365,250,110,"Read + output",["Query hidden state reads M[k]","Task loss / verified outcome"],"planned",16,13)
    d.box(880,550,250,105,"Functional update",["After the chunk reads finish","Return a new state object"],"shipped",16,13)
    d.box(500,550,270,105,"M[k+1]: next state",["Visible only to the next chunk","Commit once outside recompute"],"planned",16,13)
    d.box(60,550,300,105,"Future / target evidence",["Training labels may supervise loss","Excluded from writer + workspace"],"barrier",16,13)
    for p in [[(210,305),(210,365)],[(360,420),(500,420)],[(770,420),(880,420)],
              [(1005,305),(1005,365)],[(1005,475),(1005,550)],[(880,602),(770,602)]]:d.arrow(p)
    d.line([(210,550),(210,500)],"#B35665",2,dashed=True)
    d.text(199,489,"X",20,"#B35665",bold=True)
    d.note(500,670,630,"The fast-memory primitive is tested; its full pretrained-model online-learning integration is planned.",size=12)
    d.finish()


def learning(pdf):
    d=Drawing(pdf,5,"Autonomous improvement with independent promotion",
              "The model may propose what to learn; externally enforced budgets and acceptance rules remain separate.","05_autonomous_learning_loop")
    xs=[70,345,620,895]; width=235
    top=[("New experience",["Task outcome and evidence","Retain failures and uncertainty"]),
         ("Diagnosis",["What skill is missing?","Was the evidence sufficient?"]),
         ("Practice proposal",["Source-grounded examples","Independent correctness checks"]),
         ("Isolated candidate",["Allowed trainable groups only","Versioned parent + data lineage"])]
    for x,(title,body) in zip(xs,top):d.box(x,190,width,115,title,body,title_size=16,body_size=13)
    d.box(895,450,width,140,"Independent evaluation",["New task instances + retention","Replay-matched learning controls","Promotion set separate from audit"],title_size=15,body_size=13)
    d.box(620,450,width,140,"Decision gate",["Accept, reject or inconclusive","Hard invariants must pass","No gain means no forced update"],"shipped",16,13)
    d.box(345,450,width,140,"Versioned promotion",["Publish accepted candidate","Retain prior deployable revision","Roll back on later regression"],"shipped",16,13)
    d.box(70,450,width,140,"Later unseen tasks",["Restart before selected tests","Measure transfer and forgetting","Compare memory-only baseline"],title_size=16,body_size=13)
    for i in range(3):d.arrow([(xs[i]+width,245),(xs[i+1],245)])
    d.arrow([(1012,305),(1012,450)])
    for i in range(3,0,-1):d.arrow([(xs[i],520),(xs[i-1]+width,520)])
    d.arrow([(187,450),(187,305)])
    d.box(345,350,235,65,"Reject / inconclusive",[],"frozen",16,13)
    d.arrow([(737,450),(737,382),(580,382)])
    d.arrow([(462,350),(462,305)])
    d.note(70,632,1060,"Self-directed improvement requires unseen-task gain beyond acquisition alone. Compare autonomous target selection against a fixed verified-replay schedule.",size=15)
    d.finish()


def identity(pdf):
    d=Drawing(pdf,6,"Replica identity has a source and a separate branch",
              "Preserve source attribution while permitting a familiar parallel system to acquire its own experience.","06_replica_identity_lineage")
    d.box(60,195,290,140,"Human source archive",["Actual statements and decisions","Source spans, dates and uncertainty","No generated autobiography"])
    d.box(455,195,290,140,"Approved identity snapshot",["Selection at a documented date","Priorities, exceptions and unknowns","Versioned source-person view"],title_size=16)
    d.box(850,195,290,140,"Parallel replica",["Initialized from the snapshot","Acts and learns under delegated policy","Owns its subsequent experiences"],title_size=17,body_size=13)
    d.box(60,475,290,140,"Authorized source correction",["Human correction or changed view","Supersede affected source records","Propagate permitted invalidation"],title_size=16,body_size=13)
    d.box(455,475,290,140,"Fidelity + drift evaluation",["Decision and process agreement","Unknown-biography behavior","Traceable source / branch differences"],title_size=16,body_size=13)
    d.box(850,475,290,140,"Agent branch history",["Agent events carry agent ownership","Derived skills keep evidence lineage","Branch conclusions stay attributed"],"shipped",17,13)
    for p in [[(350,265),(455,265)],[(745,265),(850,265)],[(205,475),(205,335)],
              [(600,335),(600,475)],[(1070,335),(1070,475)],[(920,475),(920,335)],
              [(850,545),(745,545)]]:d.arrow(p)
    d.text(1085,390,"write",12,PAL["muted"])
    d.text(872,390,"read",12,PAL["muted"])
    d.line([(795,350),(795,470)],"#B35665",2,dashed=True)
    d.note(66,646,1060,"Mode is explicit: source_snapshot, source_current, parallel_replica or assistant_for_source. Agent history never silently rewrites human history.",size=14)
    d.finish()


def roadmap(pdf):
    d=Drawing(pdf,7,"Two stages: establish mechanism, then transfer scale",
              "Small-model developmental tuning and later frozen-base learning answer different experimental questions.","07_two_stage_roadmap")
    d.rect(60,160,625,505,"#FFFFFF","#CDD9E6",14,1)
    d.rect(800,160,340,505,"#FFFFFF","#CDD9E6",14,1)
    d.text(82,175,"STAGE 1 / SMALL-MODEL LAB",15,"#416EAB",bold=True)
    d.text(82,200,"MiniCPM5 text first; native V4.7 multimodal port follows",13,PAL["muted"])
    d.text(820,175,"STAGE 2 / LARGER MODELS",15,"#416EAB",bold=True)
    for x,title,body in [(85,"D0: original full",["Task training control"]),
                         (285,"D1: new modules",["Frozen base control"]),
                         (485,"D2: full augmented",["Co-adaptation test"])]:
        d.box(x,240,175,95,title,body,title_size=15,body_size=12)
    d.box(85,370,575,80,"Fair comparison + optional D3 / D4 LoRA controls",["Matched data, information, inference budget and reported training compute"],title_size=16,body_size=13)
    d.box(85,480,575,75,"Freeze a selected developmental release",["Record backbone, interfaces, tokenizer, memory format and parameter scope"],title_size=17,body_size=13)
    d.box(85,585,575,65,"Ten post-freeze learning cycles",[],title_size=17)
    for x in (172,372,572):d.arrow([(x,335),(x,370)])
    d.arrow([(372,450),(372,480)]);d.arrow([(372,555),(372,585)])
    d.box(820,240,300,110,"Per-model native integration",["Gemma 4 26B A4B","MiniCPM-V 35B A3B","Retest caches, modalities and gradients"],title_size=16,body_size=12)
    d.box(820,400,300,100,"Relearn interfaces",["Transfer records and procedures","Retrain model-specific projections"],title_size=17,body_size=13)
    d.box(820,550,300,100,"Ordinarily frozen deployment",["Promote bounded extension updates","Keep identity + retention gates"],title_size=16,body_size=13)
    d.arrow([(660,617),(746,617),(746,295),(820,295)],color="#21856C")
    d.note(690,436,50,"Evidence gate", "#21856C",12)
    d.arrow([(970,350),(970,400)]);d.arrow([(970,500),(970,550)])
    d.finish()


def data(pdf):
    d=Drawing(pdf,8,"Build data lineage before asking a teacher to expand it",
              "Public task data teaches generic mechanisms; authentic source records establish the personal replica.","08_dataset_pipeline")
    d.box(60,185,290,115,"Eligible source material",["Licensed public training material","Consented personal source records","Programmatically verifiable tasks"],title_size=17,body_size=13)
    d.box(455,185,290,115,"Assign lineage + split first",["Person / source event / task family","Freeze train, dev and audit roots","Derived variants inherit their split"],"shipped",16,13)
    d.box(850,185,290,115,"Teacher expansion",["Training roots only","Preserve sources and uncertainty","Do not invent personal memories"],title_size=17,body_size=13)
    d.box(850,415,290,135,"Independent validation",["Schema + chronological checks","Source entailment / executable oracle","Deduplication across lineage splits","Semantic verifier integration pending"],"proposed",16,12)
    d.box(455,415,290,110,"Development + calibration",["Separately validated held-out tasks","Tune settings; keep audit answers hidden"],title_size=16,body_size=13)
    d.box(455,580,290,80,"Verified training examples",["Response masks + reproducible manifests"],"shipped",16,12)
    d.box(60,415,290,135,"Sealed audit / benchmarks",["Never supplied to the teacher","No test answers in generated practice","Separate access log and final gate","Public release is not train permission"],"barrier",16,12)
    for p in [[(350,240),(455,240)],[(745,240),(850,240)],[(995,300),(995,415)],
              [(635,300),(635,415)],[(995,550),(995,620),(745,620)],
              [(555,300),(555,345),(205,345),(205,415)]]:d.arrow(p)
    d.text(648,355,"held-out dev",12,PAL["muted"])
    d.note(60,590,290,"A benchmark may inspire a new task generator. Its held-out examples and answers stay outside training.",size=14)
    d.note(60,675,1060,"Synthetic paraphrases are not independent people, independent memories or independent evaluation samples.",size=13)
    d.finish()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pdf_path=OUT/"Replica_R3_Architecture_Atlas.pdf"
    pdf=canvas.Canvas(str(pdf_path),pagesize=(W,H),pageCompression=1)
    pdf.setTitle("Replica R3 - Proposed Architecture Atlas")
    pdf.setAuthor("Replica project research and construction bundle")
    pdf.setSubject("Eight proposed architecture diagrams with implementation status boundaries")
    funcs=[system,internal,timescales,causal,learning,identity,roadmap,data]
    for fn in funcs:fn(pdf)
    pdf.save()
    if shutil.which("pdftoppm"):
        temp=OUT/"atlas_page"
        subprocess.run(["pdftoppm","-r","72","-png",str(pdf_path),str(temp)],check=True)
        slugs=["01_full_system","02_native_llama_integration","03_memory_timescales",
               "04_causal_support_query","05_autonomous_learning_loop","06_replica_identity_lineage",
               "07_two_stage_roadmap","08_dataset_pipeline"]
        for index,slug in enumerate(slugs,1):
            candidates=[OUT/f"atlas_page-{index}.png",OUT/f"atlas_page-{index:02d}.png"]
            source=next((p for p in candidates if p.exists()),None)
            if source is None:raise FileNotFoundError(candidates)
            source.replace(OUT/f"{slug}.png")
    (OUT/"README.md").write_text(
        "# Replica R3 architecture atlas\n\n"
        "Eight proposed architecture diagrams are supplied as a vector PDF, individual SVGs and PNG previews. "
        "Green identifies shipped primitives exercised by CPU contract tests; blue identifies proposed assembled integration; "
        "amber identifies optional research branches. Gray denotes native/frozen state or an update contract; red denotes a data boundary. "
        "No diagram asserts that pretrained model quality or longitudinal learning has been demonstrated.\n\n"
        "Rebuild with `python scripts/render_diagrams.py` from the project directory. Requires ReportLab; "
        "PNG rendering additionally uses `pdftoppm`.\n\n"
        "The executable text scaffold implements one native decoder insertion and prompt/source-only memory construction. "
        "Online fast-memory integration, VLM ports, independent task evaluation and the complete autonomous operating system remain milestones.\n",
        encoding="utf-8")
    print(pdf_path)


if __name__=="__main__":main()

"""
UNIFIED PDF DATA EXTRACTOR
Intelligently extracts data from multiple PDF types:
  1. Rent Roll PDFs → 2 CSVs (suites + increases)
  2. Stacking Plan PDFs → 1 CSV (suite details)

Features:
- Auto-detects PDF type (rent roll vs stacking plan)
- Handles complex layouts with spatial analysis
- Cleans vacant suite data automatically
- Zero manual configuration
"""

import pdfplumber
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PDFTypeDetector:
    """Automatically detects PDF type based on content"""
    
    @staticmethod
    def detect_pdf_type(pdf_path: str) -> str:
        """
        Analyze PDF and determine if it's a rent roll or stacking plan
        """
        with pdfplumber.open(pdf_path) as pdf:
            first_page_text = pdf.pages[0].extract_text() or ""
            
            # Rent roll indicators
            rent_roll_keywords = ['JWW001', 'RNT', 'FRE', 'CPI', 'CAM', 'TAX', 'AdditionalSpace']
            
            # Stacking plan indicators
            stacking_keywords = ['STACKING', 'RSF', 'FLOOR', 'TENANT', 'EXPIRY', 'LEASE']
            
            text_upper = first_page_text.upper()
            
            # Check for rent roll
            if any(keyword in first_page_text for keyword in rent_roll_keywords):
                return 'rent_roll'
            
            # Check for stacking plan
            if any(keyword in text_upper for keyword in stacking_keywords):
                # Additional check: stacking plans often have suite numbers like "201 -", "300 -"
                if re.search(r'\d{3,4}\s*[-–]\s*\w+', first_page_text):
                    return 'stacking_plan'
            
            return 'unknown'


class RentRollExtractor:
    """Extracts data from rent roll PDFs"""
    
    def __init__(self):
        self.suites = []
        self.increases = []
        
        # Tracking context
        self.current_suite = None
        self.current_parent_suite = None
        self.current_parent_tenant = None
        
        # Regex patterns
        self.vacant_re = re.compile(r"(JWW001\S+)\s+Vacant\s+([\d,]+)")
        self.occupied_re = re.compile(
            r"(JWW001\S+)\s+([A-Za-z0-9&\-\(\)\/\s]+?)\s+"
            r"(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}/\d{1,2}/\d{4})\s+"
            r"([\d,]+)\s*([\d,]+\.\d+)?\s*([\d\.]+)?"
        )
        self.addspace_re = re.compile(
            r"AdditionalSpace\s+JWW001-?(\S+)\s+"
            r"(\d{1,2}/\d{1,2}/\d{4})\s+"
            r"(\d{1,2}/\d{1,2}/\d{4})\s+([\d,]+)"
        )
        self.inline_increase_re = re.compile(
            r"(RNT|FRE|CPI|CAM|TAX)\s+"
            r"(\d{1,2}/\d{1,2}/\d{4})\s+"
            r"([\d,]+\.\d+)\s+"
            r"(\d+\.\d+)"
        )
    
    def extract_suite_id(self, token: str) -> str:
        """Normalize suite ID by removing JWW001 prefix"""
        token = token.replace("JWW001", "")
        token = token.lstrip("-")
        return token
    
    def extract(self, pdf_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract rent roll data from PDF"""
        
        print(f"\n    Extracting rent roll data...")
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                self._process_page_text(text)
        
        # Create DataFrames
        df_suites = pd.DataFrame(self.suites)
        df_increases = pd.DataFrame(self.increases)
        
        print(f"    Extracted {len(df_suites)} suites")
        print(f"    Extracted {len(df_increases)} rent increases")
        
        return df_suites, df_increases
    
    def _process_page_text(self, text: str):
        """Process text from a single page"""
        
        for line in text.split("\n"):
            line = line.strip()
            
            # Check for vacant suites
            if self._process_vacant_suite(line):
                continue
            
            # Check for occupied suites
            if self._process_occupied_suite(line):
                continue
            
            # Check for additional space
            if self._process_additional_space(line):
                continue
            
            # Check for inline increases
            self._process_inline_increases(line)
    
    def _process_vacant_suite(self, line: str) -> bool:
        """Process vacant suite line"""
        m = self.vacant_re.match(line)
        if m:
            full_id, sqft = m.groups()
            sid = self.extract_suite_id(full_id)
            
            self.suites.append({
                "suite": sid,
                "tenant": "",
                "rent_start": "",
                "expiry": "",
                "sqft": sqft.replace(",", ""),
                "monthly_rent": "",
                "rate_psf": "",
                "status": "Vacant"
            })
            
            self.current_suite = sid
            self.current_parent_suite = None
            self.current_parent_tenant = None
            return True
        
        return False
    
    def _process_occupied_suite(self, line: str) -> bool:
        """Process occupied suite line"""
        m = self.occupied_re.match(line)
        if m:
            full_id, tenant, start, end, sqft, monthly, psf = m.groups()
            sid = self.extract_suite_id(full_id)
            
            # Clean tenant name
            tenant = tenant.strip()
            
            self.suites.append({
                "suite": sid,
                "tenant": tenant,
                "rent_start": start,
                "expiry": end,
                "sqft": sqft.replace(",", ""),
                "monthly_rent": monthly.replace(",", "") if monthly else "",
                "rate_psf": psf if psf else "",
                "status": "Occupied"
            })
            
            self.current_suite = sid
            self.current_parent_suite = sid
            self.current_parent_tenant = tenant
            
            # Process inline increases
            for cat, date, amount, psf_val in self.inline_increase_re.findall(line):
                self.increases.append({
                    "suite": sid,
                    "tenant": tenant,
                    "category": cat,
                    "increase_date": date,
                    "monthly_amount": float(amount.replace(",", "")),
                    "psf": float(psf_val)
                })
            
            return True
        
        return False
    
    def _process_additional_space(self, line: str) -> bool:
        """Process additional space (child suite)"""
        m = self.addspace_re.match(line)
        if m:
            sid_child_raw, start, end, sqft = m.groups()
            sid_child = self.extract_suite_id(sid_child_raw)
            
            # Inherit parent tenant
            inherited_tenant = self.current_parent_tenant if self.current_parent_tenant else "AdditionalSpace"
            
            self.suites.append({
                "suite": sid_child,
                "tenant": inherited_tenant,
                "rent_start": start,
                "expiry": end,
                "sqft": sqft.replace(",", ""),
                "monthly_rent": "",
                "rate_psf": "",
                "status": "AdditionalSpace"
            })
            
            self.current_suite = sid_child
            return True
        
        return False
    
    def _process_inline_increases(self, line: str):
        """Process inline rent increases"""
        for cat, date, amount, psf_val in self.inline_increase_re.findall(line):
            if self.current_suite:
                # Find tenant for this suite
                tenant_for_inc = None
                for r in self.suites[::-1]:
                    if r["suite"] == self.current_suite:
                        tenant_for_inc = r["tenant"]
                        break
                
                self.increases.append({
                    "suite": self.current_suite,
                    "tenant": tenant_for_inc,
                    "category": cat,
                    "increase_date": date,
                    "monthly_amount": float(amount.replace(",", "")),
                    "psf": float(psf_val)
                })


class StackingPlanExtractor:
    """Extracts data from stacking plan PDFs"""
    
    def __init__(self):
        # Configuration
        self.Y_GROUP_FACTOR = 0.66
        self.LINE_LOOKAHEAD = 5
        self.X_RIGHT_ONLY = True
        
        # Regex patterns
        self.SUITE_TOKEN = r"(?:\d{3,4}|C-\d|SC-0\d)"
        self.SUITE_LINE_RE = re.compile(
            rf"(?P<suite>{self.SUITE_TOKEN})\s*[-–]\s*(?P<tenant>[^|]+?)(?=$| {self.SUITE_TOKEN}\s*[-–]|$)"
        )
        self.RSF_RE = re.compile(r"(?P<val>[\d,]{3,})\s*RSF\b", re.IGNORECASE)
        self.SF_RE = re.compile(r"(?P<val>[\d,]{3,})\s*SF\b", re.IGNORECASE)
        self.DATE_RE = re.compile(r"\b(?P<date>\d{1,2}/\d{1,2}/\d{4})\b")
        
        self.NOISE = {"REVISED", "AREAS", "OFFICE", "TOTAL", "GDSNY"}
    
    def extract(self, pdf_path: str) -> pd.DataFrame:
        """Extract stacking plan data from PDF"""
        
        print(f"\n    Extracting stacking plan data...")
        
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            words = page.extract_words(use_text_flow=True, extra_attrs=["size", "fontname"])
        
        words = [w for w in words if w.get("text") and w["text"].strip()]
        
        # Group words into lines
        lines = self._group_words_into_lines(words)
        
        # Extract suite data
        rows = self._extract_suites(lines)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        if not df.empty:
            # Clean up
            df = self._cleanup_dataframe(df)
        
        print(f"    Extracted {len(df)} suites")
        
        return df
    
    def _group_words_into_lines(self, words: List[Dict]) -> List[Dict]:
        """Group words into visual lines"""
        
        # Calculate median height
        heights = [w["bottom"] - w["top"] for w in words if w.get("top") is not None]
        med_h = sorted(heights)[len(heights) // 2] if heights else 12
        y_thresh = med_h * self.Y_GROUP_FACTOR
        
        # Add y-center and sort
        for w in words:
            w["_yc"] = (w["top"] + w["bottom"]) / 2.0
        words.sort(key=lambda w: (w["_yc"], w["x0"]))
        
        # Group into lines
        lines, cur, last_y = [], [], None
        for w in words:
            if last_y is None or abs(w["_yc"] - last_y) <= y_thresh:
                cur.append(w)
                last_y = w["_yc"]
            else:
                if cur:
                    lines.append(cur)
                cur = [w]
                last_y = w["_yc"]
        
        if cur:
            lines.append(cur)
        
        # Build line records
        out = []
        for toks in lines:
            text = " ".join(t["text"] for t in toks if t["text"])
            text = " ".join(t for t in text.split() if t not in self.NOISE)
            
            out.append({
                "y": sum(t["_yc"] for t in toks) / len(toks),
                "x": min(t["x0"] for t in toks),
                "text": text.strip(),
                "tokens": toks
            })
        
        out.sort(key=lambda r: (r["y"], r["x"]))
        return out
    
    def _extract_suites(self, lines: List[Dict]) -> List[Dict]:
        """Extract suite information from lines"""
        
        rows = []
        
        for idx, ln in enumerate(lines):
            segs = self._split_suite_segments(ln["text"])
            if not segs:
                continue
            
            # Collect areas if multiple suites on same line
            areas_for_line = None
            if len(segs) > 1:
                areas_for_line = self._collect_n_areas(lines, idx, len(segs))
            
            for k, seg in enumerate(segs):
                suite = seg["suite"]
                
                # Filter out year ranges (2027-2029, etc.)
                if self._is_year_range(suite):
                    continue
                
                tenant = self._normalize_tenant(seg["tenant"])
                
                # Find suite bbox
                suite_bbox = self._token_bbox_for_suite(ln["tokens"], suite)
                
                # Get area
                if areas_for_line is not None:
                    area = areas_for_line[k]
                else:
                    area = self._nearest_area_for_suite(lines, idx, suite_bbox)
                
                # Get expiry date
                expiry = self._find_date_down(lines, idx)
                
                # Determine status
                status = self._status_from_tenant(tenant)
                
                # CRITICAL: Clean vacant suite data
                if status == "Vacant":
                    # Only keep area/expiry if explicitly on same line
                    if not self._is_data_explicit(ln, area, expiry):
                        area = ""
                        expiry = ""
                    
                    # Special case: "Vacant (Former X)" should be just "Vacant"
                    if "former" in tenant.lower():
                        tenant = "Vacant"
                        area = ""
                        expiry = ""
                
                rows.append({
                    "suite": suite,
                    "tenant": tenant,
                    "area": area,
                    "expiry": expiry,
                    "status": status
                })
        
        return rows
    
    def _split_suite_segments(self, text: str) -> List[Dict]:
        """Split line into suite segments"""
        return [
            {"suite": m.group("suite").strip(), "tenant": m.group("tenant").strip()}
            for m in self.SUITE_LINE_RE.finditer(text)
        ]
    
    def _is_year_range(self, suite: str) -> bool:
        """Check if suite looks like a year (2027-2029, etc.)"""
        return bool(re.fullmatch(r"20[2-4]\d", suite))
    
    def _normalize_tenant(self, tenant: str) -> str:
        """Normalize tenant name"""
        if not tenant:
            return tenant
        
        # Remove RSF/SF tails
        tenant = re.sub(r"\b[\d,]+\s*(RSF|SF)\b.*$", "", tenant, flags=re.IGNORECASE)
        
        # Deduplicate words
        words = tenant.split()
        seen = []
        for w in words:
            if not seen or seen[-1].lower() != w.lower():
                seen.append(w)
        tenant = " ".join(seen)
        
        # Common corrections
        tenant = re.sub(r"\bVacamt\b", "Vacant", tenant, flags=re.IGNORECASE)
        tenant = re.sub(r"\bSecurites\b", "Securities", tenant, flags=re.IGNORECASE)
        tenant = re.sub(r"\b(LLC|Company)(\s+\1)+\b", r"\1", tenant, flags=re.IGNORECASE)
        tenant = re.sub(r"\s{2,}", " ", tenant).strip(" ,;-")
        
        return tenant
    
    def _status_from_tenant(self, tenant: str) -> str:
        """Determine occupancy status"""
        tl = (tenant or "").lower()
        if "prebuilt" in tl:
            return "Prebuilt"
        elif "vacant" in tl:
            return "Vacant"
        else:
            return "Occupied"
    
    def _token_bbox_for_suite(self, line_tokens: List, suite_text: str) -> Optional[Dict]:
        """Find token bounding box for suite"""
        s_norm = suite_text.strip()
        for t in line_tokens:
            if t["text"] and s_norm and t["text"].startswith(s_norm):
                return t
        return line_tokens[0] if line_tokens else None
    
    def _nearest_area_for_suite(self, lines: List[Dict], idx: int, suite_bbox: Optional[Dict]) -> str:
        """Find nearest area value for suite"""
        if suite_bbox is None:
            return ""
        
        def score(line, kind, token_x0, token_yc):
            same_line = (line is lines[idx])
            dy = abs(token_yc - lines[idx]["y"])
            right = (not self.X_RIGHT_ONLY) or (token_x0 >= suite_bbox["x0"])
            return (
                1 if kind == "RSF" else 0,
                1 if same_line else 0,
                1 if right else 0,
                -dy
            )
        
        best = None
        window = lines[idx:idx + self.LINE_LOOKAHEAD + 1]
        
        for ln in window:
            candidates = []
            for m in self.RSF_RE.finditer(ln["text"]):
                candidates.append(("RSF", m.group("val")))
            for m in self.SF_RE.finditer(ln["text"]):
                candidates.append(("SF", m.group("val")))
            
            for kind, val in candidates:
                tok = None
                for t in ln["tokens"]:
                    if val and val.replace(",", "") in re.sub(r"[^\d]", "", t["text"] or ""):
                        tok = t
                        break
                
                token_x0 = tok["x0"] if tok else ln["x"]
                token_yc = ((tok["top"] + tok["bottom"]) / 2.0) if tok else ln["y"]
                
                sc = score(ln, kind, token_x0, token_yc)
                if best is None or sc > best[0]:
                    best = (sc, re.sub(r"[^\d]", "", val))
        
        return best[1] if best else ""
    
    def _collect_n_areas(self, lines: List[Dict], idx: int, n: int) -> List[str]:
        """Collect n area values for multiple suites on same line"""
        found = []
        window = lines[idx:idx + self.LINE_LOOKAHEAD + 1]
        
        for ln in window:
            tmp = []
            for m in self.RSF_RE.finditer(ln["text"]):
                tmp.append(("RSF", re.sub(r"[^\d]", "", m.group("val")), m.start()))
            for m in self.SF_RE.finditer(ln["text"]):
                tmp.append(("SF", re.sub(r"[^\d]", "", m.group("val")), m.start()))
            
            tmp.sort(key=lambda x: x[2])
            
            for kind, val, _ in tmp:
                if kind == "RSF":
                    found.append(val)
                elif len(found) < n:
                    found.append(val)
            
            if len(found) >= n:
                break
        
        while len(found) < n:
            found.append("")
        
        return found[:n]
    
    def _find_date_down(self, lines: List[Dict], idx: int) -> str:
        """Find nearest date below suite"""
        window = lines[idx:idx + self.LINE_LOOKAHEAD + 1]
        best = None
        
        for ln in window:
            for m in self.DATE_RE.finditer(ln["text"]):
                dy = abs(ln["y"] - lines[idx]["y"])
                sc = -dy
                if best is None or sc > best[0]:
                    best = (sc, m.group("date"))
        
        return best[1] if best else ""
    
    def _is_data_explicit(self, line: Dict, area: str, expiry: str) -> bool:
        """Check if area/expiry are explicitly stated on the line (not inherited)"""
        line_text = line["text"]
        
        # If area/expiry appear in the same line text, they're explicit
        if area and area in line_text:
            return True
        if expiry and expiry in line_text:
            return True
        
        return False
    
    def _cleanup_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up and validate DataFrame"""
        
        # Drop duplicates
        df = df.drop_duplicates(subset=["suite", "tenant", "area", "expiry"], keep="first")
        
        # Clean tenant names
        df["tenant"] = df["tenant"].str.replace(r"\s{2,}", " ", regex=True).str.strip(" ,;-")
        
        # Guard against parking suites inheriting large office areas
        def guard_podium(suite, area):
            if suite in {"C-1", "C-6", "SC-01"} and area and int(area) > 30000:
                return ""
            return area
        
        df["area"] = df.apply(lambda r: guard_podium(r["suite"], r["area"]), axis=1)
        
        # Sort suites
        def suite_key(s):
            if isinstance(s, str) and s.startswith("SC-"):
                try:
                    return -1 + float(s.split("-")[1]) / 10.0
                except:
                    return -1.0
            if isinstance(s, str) and s.startswith("C-"):
                try:
                    return 0 + float(s.split("-")[1]) / 10.0
                except:
                    return 0.1
            try:
                return int(s)
            except:
                return 99999
        
        df = df.sort_values(by="suite", key=lambda c: c.map(suite_key)).reset_index(drop=True)
        
        return df


class UnifiedPDFExtractor:
    """
    Main unified extractor for all PDF types
    """
    
    def __init__(self, output_dir: str = './pdf_extracted'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print(" "*20 + "UNIFIED PDF DATA EXTRACTOR")
        print("="*80)
    
    def auto_discover_pdfs(self, directory: str = '.') -> List[str]:
        """Automatically discover PDF files"""
        dir_path = Path(directory)
        pdf_files = list(dir_path.glob('*.pdf'))
        return [str(f) for f in pdf_files]
    
    def process_files(self, file_paths: List[str]) -> bool:
        """Process all PDF files"""
        
        if not file_paths:
            print(" No PDF files to process")
            return False
        
        print(f"\n Discovered {len(file_paths)} PDF file(s):")
        for fp in file_paths:
            print(f"   • {Path(fp).name}")
        print()
        
        success_count = 0
        
        for file_path in file_paths:
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"[WARN]  Skipping {file_path.name} (not found)")
                continue
            
            print(f"\n{'-'*80}")
            print(f" Processing: {file_path.name}")
            print('-'*80)
            
            try:
                # Detect PDF type
                pdf_type = PDFTypeDetector.detect_pdf_type(str(file_path))
                print(f"    Detected type: {pdf_type}")
                
                if pdf_type == 'rent_roll':
                    self._process_rent_roll(file_path)
                    success_count += 1
                elif pdf_type == 'stacking_plan':
                    self._process_stacking_plan(file_path)
                    success_count += 1
                else:
                    print(f"   [WARN]  Unknown PDF type - skipping")
            
            except Exception as e:
                print(f"    Failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f" Successfully processed {success_count}/{len(file_paths)} file(s)")
        print('='*80)
        
        return success_count > 0
    
    def _process_rent_roll(self, file_path: Path):
        """Process rent roll PDF"""
        extractor = RentRollExtractor()
        df_suites, df_increases = extractor.extract(str(file_path))
        
        # Save outputs
        base_name = file_path.stem
        suites_path = self.output_dir / f'{base_name}_suites.csv'
        increases_path = self.output_dir / f'{base_name}_increases.csv'
        
        df_suites.to_csv(suites_path, index=False, encoding='utf-8-sig')
        df_increases.to_csv(increases_path, index=False, encoding='utf-8-sig')
        
        print(f"    Saved: {suites_path.name}")
        print(f"    Saved: {increases_path.name}")
    
    def _process_stacking_plan(self, file_path: Path):
        """Process stacking plan PDF"""
        extractor = StackingPlanExtractor()
        df = extractor.extract(str(file_path))
        
        # Save output
        base_name = file_path.stem
        output_path = self.output_dir / f'{base_name}_stacking_plan.csv'
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"    Saved: {output_path.name}")
    
    def run_auto(self):
        """Fully automated execution"""
        
        # Auto-discover PDFs
        files = self.auto_discover_pdfs()
        
        if not files:
            print("\n No PDF files found in current directory")
            return False
        
        # Process all files
        success = self.process_files(files)
        
        if success:
            print(f"\n All files saved to: {self.output_dir}/")
            print("\n" + "="*80)
            print(" "*25 + " EXTRACTION COMPLETE")
            print("="*80 + "\n")
        
        return success


def main():
    """Main entry point"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Unified PDF Data Extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and process all PDFs
  python pdf_extractor.py
  
  # Process specific files
  python pdf_extractor.py --files "rent_roll.pdf" "stacking_plan.pdf"
  
  # Specify output directory
  python pdf_extractor.py --output ./my_output
        """
    )
    
    parser.add_argument(
        '--files',
        nargs='+',
        help='Specific PDF files to process'
    )
    
    parser.add_argument(
        '--output',
        default='./pdf_extracted',
        help='Output directory (default: ./pdf_extracted)'
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Fully automatic mode'
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = UnifiedPDFExtractor(output_dir=args.output)
    
    # Process files
    if args.files:
        missing = [f for f in args.files if not Path(f).exists()]
        if missing:
            print(f"\n Error: Files not found:")
            for f in missing:
                print(f"   • {f}")
            sys.exit(1)
        
        success = extractor.process_files(args.files)
    else:
        if args.auto:
            success = extractor.run_auto()
        else:
            files = extractor.auto_discover_pdfs()
            if not files:
                print("\n No PDF files found")
                sys.exit(1)
            
            print(f"\n Found {len(files)} PDF(s):")
            for i, f in enumerate(files, 1):
                print(f"   {i}. {Path(f).name}")
            
            response = input("\nProcess these files? [Y/n]: ").strip().lower()
            if response in ['n', 'no']:
                sys.exit(0)
            
            success = extractor.process_files(files)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
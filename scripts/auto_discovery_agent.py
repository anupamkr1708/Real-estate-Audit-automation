"""
Automatically discovers, classifies, and routes documents to appropriate processors

Features:
- Auto-detects file types (XLS, XLSX, PDF)
- Classifies documents by content analysis
- Routes to specialized extractors
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import pdfplumber
import warnings
warnings.filterwarnings('ignore')


class IntelligentFileDiscovery:
    """
    Scans directory and intelligently classifies documents
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.classified_files = {
            'prior_ae_xls': [],
            'owner_ae_xls': [],
            'rent_roll_pdf': [],
            'stacking_plan_pdf': [],
            'unknown': []
        }
    
    def discover_all_files(self) -> Dict[str, List[str]]:
        """
        Finds and classifies ALL files
        """
        
        print("\n" + "="*80)
        print(" "*20 + " INTELLIGENT FILE DISCOVERY")
        print("="*80)
        print(f"\n Scanning directory: {self.data_dir}")
        
        if not self.data_dir.exists():
            print(f"\n[ERROR] ERROR: Directory '{self.data_dir}' does not exist!")
            return self.classified_files
        
        # Find all Excel and PDF files
        excel_files = list(self.data_dir.glob("*.xls")) + list(self.data_dir.glob("*.xlsx"))
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        print(f"\n Found {len(excel_files)} Excel file(s)")
        print(f" Found {len(pdf_files)} PDF file(s)")
        
        if not excel_files and not pdf_files:
            print("\n[WARN]  No documents found!")
            return self.classified_files
        
        # Classify files
        print("\n Classifying documents...")
        
        for xls_file in excel_files:
            self._classify_excel(xls_file)
        
        for pdf_file in pdf_files:
            self._classify_pdf(pdf_file)
        
        # Print classification results
        self._print_classification_summary()
        
        return self.classified_files
    
    def _classify_excel(self, file_path: Path):
        """
        Classify Excel file by analyzing content
        """
        
        try:
            # Read first few rows
            df = pd.read_excel(file_path, header=None, nrows=50, dtype=str)
            
            file_name = file_path.name.lower()
            content = df.to_string().lower()
            
            # Classification logic
            is_prior = any(keyword in file_name for keyword in ['prior', 'previous', 'old', '142'])
            is_owner = any(keyword in file_name for keyword in ['owner', 'current', 'new'])
            
            # Content-based detection
            has_lease_structure = any(pattern in content for pattern in [
                'suite:', 'tenant', 'lease', 'rent', 'rsf', 'area'
            ])
            
            has_numbering = bool(re.search(r'\d+\.\s+[A-Za-z]', content))
            
            if has_lease_structure and has_numbering:
                if is_prior or 'prior' in content:
                    self.classified_files['prior_ae_xls'].append(str(file_path))
                    print(f"  [OK] {file_path.name} → Prior AE (Lease Master)")
                elif is_owner or 'owner' in content:
                    self.classified_files['owner_ae_xls'].append(str(file_path))
                    print(f"  [OK] {file_path.name} → Owner AE (Lease Master)")
                else:
                    # Default: if unclear, check which type is missing
                    if not self.classified_files['prior_ae_xls']:
                        self.classified_files['prior_ae_xls'].append(str(file_path))
                        print(f"  [OK] {file_path.name} → Prior AE (auto-assigned)")
                    else:
                        self.classified_files['owner_ae_xls'].append(str(file_path))
                        print(f"  [OK] {file_path.name} → Owner AE (auto-assigned)")
            else:
                self.classified_files['unknown'].append(str(file_path))
                print(f"  [WARN]  {file_path.name} → Unknown format")
        
        except Exception as e:
            print(f"  [ERROR] {file_path.name} → Error reading: {e}")
            self.classified_files['unknown'].append(str(file_path))
    
    def _classify_pdf(self, file_path: Path):
        """
        Classify PDF file by analyzing content with improved logic
        """
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Analyze first page
                first_page_text = pdf.pages[0].extract_text() or ""
                
                # Also check second page for better classification
                second_page_text = ""
                if len(pdf.pages) > 1:
                    second_page_text = pdf.pages[1].extract_text() or ""
                
                combined_text = (first_page_text + " " + second_page_text).upper()
                file_name = file_path.name.lower()
                
                # PRIORITY 1: Strong Stacking Plan indicators (check first)
                stacking_strong_indicators = [
                    'STACKING PLAN',
                    'FLOOR PLAN', 
                    'SUITE STACKING',
                    'BUILDING STACKING'
                ]
                
                stacking_pattern_indicators = [
                    bool(re.search(r'\d{3,4}\s*[-–]\s+[A-Z]', first_page_text)),  # "201 - Tenant"
                    bool(re.search(r'\d+\s+RSF', combined_text)),  # "5,000 RSF"
                    bool(re.search(r'FLOOR\s+\d+', combined_text)),  # "FLOOR 2"
                    ('EXPIRY' in combined_text or 'EXPIRATION' in combined_text) and 'RSF' in combined_text
                ]
                
                is_strong_stacking = (
                    any(indicator in combined_text for indicator in stacking_strong_indicators) or
                    sum(stacking_pattern_indicators) >= 2  # At least 2 pattern matches
                )
                
                # PRIORITY 2: Strong Rent Roll indicators
                rent_roll_strong_indicators = [
                    'JWW001',  # Specific suite ID pattern
                    'ADDITIONAL SPACE',  # Rent roll specific
                    'RNT' in combined_text and 'FRE' in combined_text,  # Multiple rent codes
                ]
                
                rent_roll_keywords = ['RENT ROLL', 'TENANT SCHEDULE', 'MONTHLY RENT']
                
                is_strong_rent_roll = (
                    any(indicator in combined_text for indicator in rent_roll_keywords) or
                    any(rent_roll_strong_indicators)
                )
                
                # Classification logic with priority
                if is_strong_stacking and not is_strong_rent_roll:
                    self.classified_files['stacking_plan_pdf'].append(str(file_path))
                    print(f"  [OK] {file_path.name} -> Stacking Plan")
                elif is_strong_rent_roll and not is_strong_stacking:
                    self.classified_files['rent_roll_pdf'].append(str(file_path))
                    print(f"  [OK] {file_path.name} -> Rent Roll")
                elif is_strong_stacking:  # Both detected, but stacking stronger
                    self.classified_files['stacking_plan_pdf'].append(str(file_path))
                    print(f"  [OK] {file_path.name} -> Stacking Plan (primary)")
                elif is_strong_rent_roll:
                    self.classified_files['rent_roll_pdf'].append(str(file_path))
                    print(f"  [OK] {file_path.name} -> Rent Roll")
                else:
                    # Fallback: check filename
                    if 'stack' in file_name or 'floor' in file_name:
                        self.classified_files['stacking_plan_pdf'].append(str(file_path))
                        print(f"  [OK] {file_path.name} -> Stacking Plan (by filename)")
                    elif 'rent' in file_name and 'roll' in file_name:
                        self.classified_files['rent_roll_pdf'].append(str(file_path))
                        print(f"  [OK] {file_path.name} -> Rent Roll (by filename)")
                    else:
                        self.classified_files['unknown'].append(str(file_path))
                        print(f"  [WARN] {file_path.name} -> Unknown PDF type")
        
        except Exception as e:
            print(f"  [ERROR] {file_path.name} → Error reading: {e}")
            self.classified_files['unknown'].append(str(file_path))
    
    def _print_classification_summary(self):
        """
        Print summary of classification results
        """
        
        print("\n" + "="*80)
        print(" CLASSIFICATION SUMMARY")
        print("="*80)
        
        total_classified = sum(len(v) for k, v in self.classified_files.items() if k != 'unknown')
        total_unknown = len(self.classified_files['unknown'])
        
        print(f"\n[OK] Successfully classified: {total_classified} file(s)")
        
        for category, files in self.classified_files.items():
            if files and category != 'unknown':
                print(f"  • {category.replace('_', ' ').title()}: {len(files)}")
                for f in files:
                    print(f"    - {Path(f).name}")
        
        if total_unknown > 0:
            print(f"\n[WARN]  Unknown files: {total_unknown}")
            for f in self.classified_files['unknown']:
                print(f"    - {Path(f).name}")
        
        print("="*80)
    
    def get_processing_plan(self) -> Dict[str, any]:
        """
        Generate processing plan based on discovered files
        """
        
        plan = {
            'has_prior_ae': len(self.classified_files['prior_ae_xls']) > 0,
            'has_owner_ae': len(self.classified_files['owner_ae_xls']) > 0,
            'has_rent_roll': len(self.classified_files['rent_roll_pdf']) > 0,
            'has_stacking_plan': len(self.classified_files['stacking_plan_pdf']) > 0,
            'can_proceed': False,
            'missing': [],
            'files': self.classified_files
        }
        
        # Check minimum requirements (at least 2 sources)
        sources = sum([
            plan['has_prior_ae'],
            plan['has_owner_ae'],
            plan['has_rent_roll'],
            plan['has_stacking_plan']
        ])
        
        if sources >= 2:
            plan['can_proceed'] = True
        else:
            if not plan['has_prior_ae']:
                plan['missing'].append('Prior AE')
            if not plan['has_owner_ae']:
                plan['missing'].append('Owner AE')
            if not plan['has_rent_roll']:
                plan['missing'].append('Rent Roll')
            if not plan['has_stacking_plan']:
                plan['missing'].append('Stacking Plan')
        
        return plan


def main():
    """
    Standalone testing
    """
    
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    
    agent = IntelligentFileDiscovery(data_dir)
    classified = agent.discover_all_files()
    
    plan = agent.get_processing_plan()
    
    print("\n" + "="*80)
    print(" PROCESSING PLAN")
    print("="*80)
    
    if plan['can_proceed']:
        print("\n Ready to process!")
        print(f"  • Found {sum(len(v) for k, v in classified.items() if k != 'unknown')} valid documents")
    else:
        print("\n Cannot proceed - insufficient data sources")
        print(f"  • Need at least 2 sources")
        print(f"  • Missing: {', '.join(plan['missing'])}")
    
    print("="*80)


if __name__ == "__main__":
    main()
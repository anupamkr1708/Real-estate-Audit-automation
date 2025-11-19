#!/usr/bin/env python3
"""
Fully automated end-to-end processing of ANY lease documents

Usage:
    python master_orchestrator.py
    python master_orchestrator.py --data-dir /path/to/documents
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Windows-safe logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("audit_automation.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("orchestrator")


class AgenticAuditOrchestrator:
    """
    Master orchestrator that manages entire audit workflow
    """
    
    def __init__(self, data_dir: str = "./data", output_dir: str = "./results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create directory structure
        self.dirs = {
            'extracted_xls': self.output_dir / 'extracted_xls',
            'extracted_pdf': self.output_dir / 'extracted_pdf',
            'validated': self.output_dir / 'validated',
            'findings': self.output_dir / 'findings',
            'reports': self.output_dir / 'reports'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.classified_files = {}
        self.processing_log = []
    
    def run_full_pipeline(self) -> bool:
        """
        Execute complete automated pipeline
        """
        
        print("\n" + "="*100)
        print(" "*35 + "AGENTIC AUDIT ORCHESTRATOR")
        print("="*100)
        print(f"\nData Directory: {self.data_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        try:
            # PHASE 1: Discovery
            if not self.phase_1_discovery():
                return False
            
            # PHASE 2: Extraction
            if not self.phase_2_extraction():
                return False
            
            # PHASE 3: Validation
            if not self.phase_3_validation():
                return False
            
            # PHASE 4: Calculation
            if not self.phase_4_calculation():
                return False
            
            # PHASE 5: Report Generation
            if not self.phase_5_reporting():
                return False
            
            # Success!
            self.print_success_summary()
            return True
        
        except Exception as e:
            logger.error(f" Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase_1_discovery(self) -> bool:
        """
        PHASE 1: Intelligent File Discovery
        """
        
        print("\n" + ">"*100)
        print("PHASE 1: INTELLIGENT FILE DISCOVERY")
        print(">"*100)
        
        self.log_phase("PHASE 1: Discovery")
        
        try:
            # Import discovery agent
            from auto_discovery_agent import IntelligentFileDiscovery
            
            agent = IntelligentFileDiscovery(str(self.data_dir))
            self.classified_files = agent.discover_all_files()
            
            plan = agent.get_processing_plan()
            
            if not plan['can_proceed']:
                logger.error(" Insufficient data sources!")
                logger.error(f"   Missing: {', '.join(plan['missing'])}")
                logger.error("   Need at least 2 data sources to proceed")
                return False
            
            logger.info("Discovery complete - ready to process")
            self.log_success("Discovery successful")
            return True
        
        except ImportError:
            logger.error(" auto_discovery_agent.py not found!")
            logger.error("   Make sure all required files are in the same directory")
            return False
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            return False
    
    def phase_2_extraction(self) -> bool:
        """
        PHASE 2: Extract data from all documents
        """
        
        print("\n" + ">"*100)
        print("PHASE 2: DATA EXTRACTION")
        print(">"*100)
        
        self.log_phase("PHASE 2: Extraction")
        
        # Extract XLS files
        if self.classified_files.get('prior_ae_xls') or self.classified_files.get('owner_ae_xls'):
            logger.info("\n[*] Extracting Excel files...")
            if not self._extract_excel_files():
                return False
        
        # Extract PDF files
        if self.classified_files.get('rent_roll_pdf') or self.classified_files.get('stacking_plan_pdf'):
            logger.info("\n[*] Extracting PDF files...")
            if not self._extract_pdf_files():
                return False
        
        logger.info("Extraction complete")
        self.log_success("Extraction successful")
        return True
    
    def _extract_excel_files(self) -> bool:
        """
        Extract data from Excel files
        """
        
        try:
            if not Path("combined_extractor.py").exists():
                logger.error("combined_extractor.py not found!")
                return False
            
            all_xls = (
                self.classified_files.get('prior_ae_xls', []) +
                self.classified_files.get('owner_ae_xls', [])
            )
            
            if not all_xls:
                logger.warning(" No Excel files to extract")
                return True
            
            cmd = [
                sys.executable, "combined_extractor.py",
                "--files"] + all_xls + [
                "--output", str(self.dirs['extracted_xls']),
                "--auto"
            ]
            
            logger.info(f"   Running: combined_extractor.py")
            
            # Set UTF-8 encoding for subprocess
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace', env=env)
            
            if result.returncode != 0:
                logger.error(" Excel extraction failed!")
                logger.error(result.stderr)
                return False
            
            logger.info("   Excel extraction successful")
            return True
        
        except Exception as e:
            logger.error(f"Excel extraction error: {e}")
            return False
    
    def _extract_pdf_files(self) -> bool:
        """
        Extract data from PDF files
        """
        
        try:
            if not Path("pdf_extractor.py").exists():
                logger.error(" pdf_extractor.py not found!")
                return False
            
            all_pdfs = (
                self.classified_files.get('rent_roll_pdf', []) +
                self.classified_files.get('stacking_plan_pdf', [])
            )
            
            if not all_pdfs:
                logger.warning(" No PDF files to extract")
                return True
            
            cmd = [
                sys.executable, "pdf_extractor.py",
                "--files"] + all_pdfs + [
                "--output", str(self.dirs['extracted_pdf']),
                "--auto"
            ]
            
            logger.info(f"   Running: pdf_extractor.py")
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding='utf-8', errors='replace', env=env)
            
            if result.returncode != 0:
                logger.error(" PDF extraction failed!")
                logger.error(result.stderr)
                return False
            
            logger.info("    PDF extraction successful")
            return True
        
        except Exception as e:
            logger.error(f" PDF extraction error: {e}")
            return False
    
    def phase_3_validation(self) -> bool:
        """
        PHASE 3: Validate and standardize all extracted data
        """
        
        print("\n" + ">"*100)
        print("PHASE 3: DATA VALIDATION")
        print(">"*100)
        
        self.log_phase("PHASE 3: Validation")
        
        try:
            if not Path("validator.py").exists():
                logger.error(" validator.py not found!")
                return False
            
            cmd = [
                sys.executable, "validator.py",
                "--xls-input", str(self.dirs['extracted_xls']),
                "--pdf-input", str(self.dirs['extracted_pdf']),
                "--output", str(self.dirs['validated'])
            ]
            
            logger.info("   Running: validator.py")
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding='utf-8', errors='replace', env=env)
            
            if result.returncode != 0:
                logger.error("[ERROR] Validation failed!")
                logger.error(result.stderr)
                return False
            
            logger.info(" Validation complete")
            self.log_success("Validation successful")
            return True
        
        except Exception as e:
            logger.error(f" Validation error: {e}")
            return False
    
    def phase_4_calculation(self) -> bool:
        """
        PHASE 4: Calculate discrepancies and findings
        """
        
        print("\n" + ">"*100)
        print("PHASE 4: DISCREPANCY CALCULATION")
        print(">"*100)
        
        self.log_phase("PHASE 4: Calculation")
        
        try:
            if not Path("calculator.py").exists():
                logger.error(" calculator.py not found!")
                return False
            
            cmd = [
                sys.executable, "calculator.py",
                "--input", str(self.dirs['validated']),
                "--output", str(self.dirs['findings'])
            ]
            
            logger.info("   Running: calculator.py")
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding='utf-8', errors='replace', env=env)
            
            if result.returncode != 0:
                logger.error(" Calculation failed!")
                logger.error(result.stderr)
                return False
            
            logger.info("Calculation complete")
            self.log_success("Calculation successful")
            return True
        
        except Exception as e:
            logger.error(f" Calculation error: {e}")
            return False
    
    def phase_5_reporting(self) -> bool:
        """
        PHASE 5: Generate final audit report
        """
        
        print("\n" + ">"*100)
        print("PHASE 5: REPORT GENERATION")
        print(">"*100)
        
        self.log_phase("PHASE 5: Reporting")
        
        try:
            if not Path("reporter.py").exists():
                logger.error(" reporter.py not found!")
                return False
            
            report_path = self.dirs['reports'] / f"Audit_Report_{self.timestamp}.xlsx"
            
            cmd = [
                sys.executable, "reporter.py",
                "--input", str(self.dirs['findings']),
                "--output", str(report_path)
            ]
            
            logger.info("   Running: reporter.py")
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding='utf-8', errors='replace', env=env)
            
            if result.returncode != 0:
                logger.error(" Report generation failed!")
                logger.error(result.stderr)
                return False
            
            self.final_report_path = report_path
            logger.info(" Report generation complete")
            self.log_success("Report generated successfully")
            return True
        
        except Exception as e:
            logger.error(f" Report generation error: {e}")
            return False
    
    def log_phase(self, phase_name: str):
        """Log phase start"""
        self.processing_log.append({
            'phase': phase_name,
            'timestamp': datetime.now(),
            'status': 'started'
        })
    
    def log_success(self, message: str):
        """Log success"""
        if self.processing_log:
            self.processing_log[-1]['status'] = 'success'
            self.processing_log[-1]['message'] = message
    
    def print_success_summary(self):
        """
        Print final success summary
        """
        
        print("\n" + "="*100)
        print(" "*35 + " PIPELINE COMPLETED SUCCESSFULLY")
        print("="*100)
        
        print("\nPROCESSING SUMMARY:")
        for log_entry in self.processing_log:
            status_icon = "[OK]" if log_entry['status'] == 'success' else "[*]"
            print(f"   {status_icon} {log_entry['phase']}: {log_entry.get('message', 'Complete')}")
        
        print(f"\nFINAL AUDIT REPORT:")
        print(f"   Location: {self.final_report_path}")
        print(f"   Size: {self.final_report_path.stat().st_size / 1024:.1f} KB")
        
        print(f"\nALL OUTPUT DIRECTORIES:")
        for name, path in self.dirs.items():
            file_count = len(list(path.glob("*")))
            print(f"   * {name}: {path} ({file_count} files)")
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Full log: audit_automation.log")
        print("="*100 + "\n")


def main():
    """
    Main entry point
    """
    
    parser = argparse.ArgumentParser(
        description='Agentic Audit Automation - Fully Automated Pipeline'
    )
    
    parser.add_argument(
        '--data-dir',
        default='./data',
        help='Directory containing lease documents (default: ./data)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./results',
        help='Output directory for all results (default: ./results)'
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = AgenticAuditOrchestrator(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    success = orchestrator.run_full_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
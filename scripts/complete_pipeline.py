#!/usr/bin/env python3
"""
USAGE:
    python complete_pipeline.py
OR
    python complete_pipeline.py --validated validated_data --findings findings --reports final_reports
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("pipeline")


class FinalAuditPipeline:

    def __init__(self, validated_dir, findings_dir, reports_dir):
        self.validated = Path(validated_dir)
        self.findings = Path(findings_dir)
        self.reports = Path(reports_dir)

        # Create folders if missing
        self.validated.mkdir(exist_ok=True)
        self.findings.mkdir(exist_ok=True)
        self.reports.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_path = self.reports / f"Audit_Report_{timestamp}.xlsx"

        logger.info(f"Validated folder : {self.validated}")
        logger.info(f"Findings folder  : {self.findings}")
        logger.info(f"Reports folder   : {self.reports}")

    # -------------------------------------------------------------
    #  PHASE 1 – VALIDATION
    # -------------------------------------------------------------
    def run_validation(self):
        logger.info("\n=============================")
        logger.info(" PHASE 1: VALIDATION")
        logger.info("=============================\n")

        if not Path("validator.py").exists():
            logger.error(" validator.py not found in parent directory!")
            return False

        cmd = [
            sys.executable, "validator.py",
            "--xls-input", "extracted_data",          # fixed based on your earlier message
            "--pdf-input", "pdf_extracted",
            "--output", str(self.validated)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(" VALIDATION FAILED")
            logger.error(result.stderr)
            return False

        logger.info(" Validation Completed Successfully")
        return True

    # -------------------------------------------------------------
    #  PHASE 2 – CALCULATION
    # -------------------------------------------------------------
    def run_calculator(self):
        logger.info("\n=============================")
        logger.info(" PHASE 2: CALCULATOR")
        logger.info("=============================\n")

        if not Path("calculator.py").exists():
            logger.error(" calculator.py not found!")
            return False

        cmd = [
            sys.executable, "calculator.py",
            "--input", str(self.validated),
            "--output", str(self.findings)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(" CALCULATOR FAILED")
            logger.error(result.stderr)
            return False

        logger.info(" Calculator Completed Successfully")
        return True

    # -------------------------------------------------------------
    #  PHASE 3 – REPORT GENERATION
    # -------------------------------------------------------------
    def run_reporter(self):
        logger.info("\n=============================")
        logger.info(" PHASE 3: REPORT GENERATION")
        logger.info("=============================\n")

        if not Path("reporter.py").exists():
            logger.error(" reporter.py not found!")
            return False

        cmd = [
            sys.executable, "reporter.py",
            "--input", str(self.findings),
            "--output", str(self.report_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(" REPORT GENERATION FAILED")
            logger.error(result.stderr)
            return False

        logger.info(f" Report generated: {self.report_path}")
        return True

    # -------------------------------------------------------------
    #  RUN FULL PIPELINE
    # -------------------------------------------------------------
    def run(self):

        logger.info("\n====================================================")
        logger.info("            STARTING FULL AUDIT PIPELINE")
        logger.info("====================================================")

        # PHASE 1
        if not self.run_validation():
            return False

        # PHASE 2
        if not self.run_calculator():
            return False

        # PHASE 3
        if not self.run_reporter():
            return False

        logger.info("\n====================================================")
        logger.info("               PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("====================================================")
        logger.info(f"FINAL REPORT AVAILABLE AT:\n{self.report_path}\n")

        return True


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--validated", default="validated_data", help="Folder containing validated CSVs")
    parser.add_argument("--findings", default="findings", help="Folder to store calculator results")
    parser.add_argument("--reports", default="final_reports", help="Where to store the final Excel report")
    args = parser.parse_args()

    pipeline = FinalAuditPipeline(
        validated_dir=args.validated,
        findings_dir=args.findings,
        reports_dir=args.reports
    )

    pipeline.run()


if __name__ == "__main__":
    main()

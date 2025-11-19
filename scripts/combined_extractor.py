"""
Fully automated XLS extractor for lease data

OUTPUT: For each input file, generates:
  1. {filename}_lease_master.csv 
  2. {filename}_rent_escalations.csv

Features:
- Auto-detects files in current directory
- Intelligent column detection
- Handles variable row structures
- Works with any similar lease format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class IntelligentColumnDetector:
    """
    Automatically detects column positions
    Analyzes content patterns to identify data types
    """
    
    @staticmethod
    def detect_columns(df: pd.DataFrame, sample_size: int = 30) -> Dict[str, int]:
        """
        Analyze DataFrame and return detected column positions
        """
        
        sample = df.head(sample_size)
        detected = {}
        
        # Column 0: Always tenant info (numbered entries like "1. Tenant")
        detected['tenant_info'] = 0
        
        # Column 1: Area and building share (numeric, typically 100-50000 range)
        detected['area'] = 1
        
        # Column 2: Lease type/space category (text: "Base", "Contract", "OFFICE")
        detected['lease_type'] = 2
        
        # Column 3: Rates and amounts (numeric, various ranges)
        detected['rates'] = 3
        
        # Column 4-6: Escalation dates and rates (contains dates and numbers)
        detected['escalation_dates'] = IntelligentColumnDetector._find_date_column(sample, start_col=4)
        detected['escalation_annual'] = detected['escalation_dates'] + 1 if detected['escalation_dates'] else 5
        detected['escalation_monthly'] = detected['escalation_dates'] + 2 if detected['escalation_dates'] else 6
        
        # Column 8-9: Free rent (date and "X Months" text)
        detected['free_rent_date'] = IntelligentColumnDetector._find_date_column(sample, start_col=8, max_col=10)
        detected['free_rent_months'] = IntelligentColumnDetector._find_months_column(sample, start_col=8)
        
        # Column 11: Market renewal (text containing "Market - XX%")
        detected['market_renewal'] = IntelligentColumnDetector._find_pattern_column(sample, r'Market.*\d+%')
        
        # Column 12: Recovery structure (text like "GLOBAL OFFICE", "HILTON")
        detected['recovery'] = IntelligentColumnDetector._find_text_column(sample, min_col=12, max_col=15)
        
        # Column 14-15: TI amounts (numeric)
        detected['ti'] = IntelligentColumnDetector._find_numeric_column(sample, start_col=13, max_col=16)
        
        # Column 17: Commission (numeric)
        detected['commission'] = IntelligentColumnDetector._find_numeric_column(sample, start_col=16, max_col=20)
        
        return detected
    
    @staticmethod
    def _find_date_column(sample: pd.DataFrame, start_col: int = 0, max_col: int = None) -> Optional[int]:
        """Find column containing date patterns"""
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'[A-Z][a-z]{2}-\d{4}',     # Mon-YYYY
            r'\d{4}-\d{2}-\d{2}'        # YYYY-MM-DD
        ]
        
        max_col = max_col or len(sample.columns)
        
        for col_idx in range(start_col, min(max_col, len(sample.columns))):
            date_count = 0
            for val in sample.iloc[:, col_idx]:
                if pd.notna(val) and any(re.search(p, str(val)) for p in date_patterns):
                    date_count += 1
            
            if date_count >= 3:
                return col_idx
        
        return start_col
    
    @staticmethod
    def _find_months_column(sample: pd.DataFrame, start_col: int = 0) -> Optional[int]:
        """Find column containing 'X Months' pattern"""
        for col_idx in range(start_col, min(start_col + 5, len(sample.columns))):
            for val in sample.iloc[:, col_idx]:
                if pd.notna(val) and re.search(r'\d+\.?\d*\s*Months?', str(val), re.IGNORECASE):
                    return col_idx
        
        return start_col + 1
    
    @staticmethod
    def _find_pattern_column(sample: pd.DataFrame, pattern: str) -> Optional[int]:
        """Find column matching a regex pattern"""
        for col_idx in range(len(sample.columns)):
            for val in sample.iloc[:, col_idx]:
                if pd.notna(val) and re.search(pattern, str(val), re.IGNORECASE):
                    return col_idx
        
        return 11  # Default fallback
    
    @staticmethod
    def _find_text_column(sample: pd.DataFrame, min_col: int, max_col: int) -> Optional[int]:
        """Find column with substantial text content"""
        for col_idx in range(min_col, min(max_col, len(sample.columns))):
            text_count = 0
            for val in sample.iloc[:, col_idx]:
                if pd.notna(val):
                    val_str = str(val).strip()
                    if len(val_str) > 3 and val_str not in ['nan', 'None']:
                        text_count += 1
            
            if text_count >= 5:
                return col_idx
        
        return min_col
    
    @staticmethod
    def _find_numeric_column(sample: pd.DataFrame, start_col: int, max_col: int) -> Optional[int]:
        """Find column with numeric values"""
        for col_idx in range(start_col, min(max_col, len(sample.columns))):
            numeric_count = 0
            for val in sample.iloc[:, col_idx]:
                try:
                    if pd.notna(val):
                        float(str(val).replace(',', '').replace('$', ''))
                        numeric_count += 1
                except:
                    pass
            
            if numeric_count >= 3:
                return col_idx
        
        return start_col


class SmartParser:
    """
    Intelligent parsing with automatic format detection
    """
    
    @staticmethod
    def parse_date(value: str) -> Optional[str]:
        """Auto-detect and parse date format"""
        if pd.isna(value) or str(value).strip() in ['', 'nan', 'None', 'N/A']:
            return None
        
        date_str = str(value).strip()
        
        # Try multiple formats automatically
        formats = [
            ('%m/%d/%Y', '%Y-%m-%d'),
            ('%d/%m/%Y', '%Y-%m-%d'),
            ('%b-%Y', '%Y-%m-01'),
            ('%B-%Y', '%Y-%m-01'),
            ('%Y-%m-%d', '%Y-%m-%d'),
            ('%m-%d-%Y', '%Y-%m-%d'),
            ('%Y/%m/%d', '%Y-%m-%d'),
        ]
        
        for in_fmt, out_fmt in formats:
            try:
                dt = datetime.strptime(date_str, in_fmt)
                return dt.strftime(out_fmt)
            except:
                continue
        
        return None
    
    @staticmethod
    def parse_number(value: str, allow_zero: bool = True) -> Optional[float]:
        """Parse numeric value with automatic cleaning"""
        if pd.isna(value) or str(value).strip() in ['', 'nan', 'None', 'N/A']:
            return None
        
        value_str = str(value).strip()
        
        # Remove common formatting
        value_str = value_str.replace('$', '').replace(',', '').replace('%', '')
        
        if value_str in ['0', '0.0'] and allow_zero:
            return 0.0
        
        try:
            return float(value_str)
        except:
            return None
    
    @staticmethod
    def split_delimited(value: str, delimiter: str = '|') -> List[str]:
        """Split and clean delimited values"""
        if pd.isna(value) or str(value).strip() in ['', 'nan', 'None']:
            return []
        
        parts = str(value).split(delimiter)
        return [p.strip() for p in parts if p.strip() and p.strip() != 'nan']
    
    @staticmethod
    def extract_pattern(value: str, pattern: str, group: int = 1) -> Optional[str]:
        """Extract value using regex pattern"""
        if pd.isna(value):
            return None
        
        match = re.search(pattern, str(value), re.IGNORECASE)
        return match.group(group) if match else None


class IntelligentLeaseExtractor:
    """
    Fully automated lease data extractor
    Zero configuration required
    """
    
    def __init__(self, output_dir: str = './extracted_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}  # {filename: {lease_master: df, rent_escalations: df}}
        
        print("\n" + "="*80)
        print(" "*20 + "INTELLIGENT LEASE DATA EXTRACTOR")
        print("="*80)
    
    def auto_discover_files(self, directory: str = '.') -> List[str]:
        """Automatically discover XLS/XLSX files in directory"""
        
        dir_path = Path(directory)
        
        # Find all Excel files
        xls_files = list(dir_path.glob('*.xls')) + list(dir_path.glob('*.xlsx'))
        
        # Filter for lease-related files (common naming patterns)
        lease_keywords = ['ae', 'lease', 'rent', 'tenant', 'prior', 'owner', 'current']
        
        relevant_files = []
        for file in xls_files:
            file_lower = file.name.lower()
            if any(keyword in file_lower for keyword in lease_keywords):
                relevant_files.append(str(file))
        
        return relevant_files
    
    def process_files(self, file_paths: List[str]) -> bool:
        """Process multiple files automatically"""
        
        if not file_paths:
            print(" No files to process")
            return False
        
        print(f"\n Discovered {len(file_paths)} file(s):")
        for fp in file_paths:
            print(f"   • {Path(fp).name}")
        print()
        
        success_count = 0
        
        for file_path in file_paths:
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"  Skipping {file_path.name} (not found)")
                continue
            
            print(f"\n{'-'*80}")
            print(f" Processing: {file_path.name}")
            print('-'*80)
            
            try:
                # Extract data
                lease_master, rent_escalations = self._process_single_file(file_path)
                
                # Store results
                self.results[file_path.stem] = {
                    'lease_master': lease_master,
                    'rent_escalations': rent_escalations
                }
                
                success_count += 1
                
                print(f" Successfully extracted:")
                print(f"   • {len(lease_master)} leases")
                print(f"   • {len(rent_escalations)} rent escalations")
                
            except Exception as e:
                print(f" Failed to process {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f" Successfully processed {success_count}/{len(file_paths)} file(s)")
        print('='*80)
        
        return success_count > 0
    
    def _process_single_file(self, file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process a single lease file"""
        
        # Step 1: Read file
        df = self._read_excel_file(file_path)
        print(f"   Loaded: {len(df)} rows × {len(df.columns)} columns")
        
        # Step 2: Auto-detect columns
        print(f"    Auto-detecting column structure...")
        column_map = IntelligentColumnDetector.detect_columns(df)
        print(f"    Detected {len(column_map)} column types")
        
        # Step 3: Find data boundaries
        start_row = self._find_data_start(df)
        print(f"    Data starts at row {start_row}")
        
        # Step 4: Identify tenant groups
        tenant_groups = self._identify_tenant_groups(df, start_row)
        print(f"    Identified {len(tenant_groups)} tenant groups")
        
        # Step 5: Extract data
        print(f"    Extracting lease data...")
        lease_master = self._extract_lease_master(tenant_groups, column_map)
        
        print(f"    Extracting escalations...")
        rent_escalations = self._extract_rent_escalations(tenant_groups, column_map)
        
        return lease_master, rent_escalations
    
    def _read_excel_file(self, file_path: Path) -> pd.DataFrame:
        """Read Excel file with automatic engine selection"""
        
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.xls':
            try:
                # Try xlrd first
                import xlrd
                workbook = xlrd.open_workbook(file_path)
                sheet = workbook.sheet_by_index(0)
                
                data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] 
                        for r in range(sheet.nrows)]
                
                return pd.DataFrame(data)
            except:
                # Fallback to pandas
                return pd.read_excel(file_path, header=None, dtype=str)
        else:
            return pd.read_excel(file_path, engine='openpyxl', header=None, dtype=str)
    
    def _find_data_start(self, df: pd.DataFrame) -> int:
        """Find where tenant data begins"""
        
        # Look for numbered tenant entries (e.g., "1. Tenant Name")
        for idx in range(min(100, len(df))):  # Check first 100 rows
            first_col = str(df.iloc[idx, 0]) if pd.notna(df.iloc[idx, 0]) else ''
            if re.match(r'^\d+\.\s+', first_col):
                return idx
        
        return 0
    
    def _identify_tenant_groups(self, df: pd.DataFrame, start_row: int) -> List[pd.DataFrame]:
        """Identify variable-length tenant record groups"""
        
        groups = []
        current_row = start_row
        
        while current_row < len(df):
            first_col = str(df.iloc[current_row, 0]) if pd.notna(df.iloc[current_row, 0]) else ''
            
            if re.match(r'^\d+\.\s+', first_col):
                # Found a tenant - find where it ends
                next_tenant = current_row + 1
                
                for check_row in range(current_row + 1, len(df)):
                    check_col = str(df.iloc[check_row, 0]) if pd.notna(df.iloc[check_row, 0]) else ''
                    if re.match(r'^\d+\.\s+', check_col):
                        next_tenant = check_row
                        break
                else:
                    next_tenant = len(df)
                
                group = df.iloc[current_row:next_tenant].copy()
                groups.append(group)
                current_row = next_tenant
            else:
                current_row += 1
        
        return groups
    
    def _extract_lease_master(self, tenant_groups: List[pd.DataFrame], 
                               column_map: Dict) -> pd.DataFrame:
        """Extract all lease master data"""
        
        records = []
        
        for group in tenant_groups:
            try:
                record = self._extract_single_lease(group, column_map)
                records.append(record)
            except Exception as e:
                print(f"        Warning: Failed to extract lease: {e}")
                continue
        
        return pd.DataFrame(records)
    
    def _extract_single_lease(self, group: pd.DataFrame, column_map: Dict) -> Dict:
        """Extract data from a single tenant group"""
        
        # Row 1: Tenant name and number
        row1 = str(group.iloc[0, 0]) if pd.notna(group.iloc[0, 0]) else ''
        tenant_match = re.match(r'^(\d+)\.\s+(.+)', row1)
        
        tenant_number = int(tenant_match.group(1)) if tenant_match else None
        tenant_name_raw = tenant_match.group(2).strip() if tenant_match else None
        is_speculative = '**To-Be-Leased' in str(tenant_name_raw) if tenant_name_raw else False
        tenant_name = tenant_name_raw.replace('**To-Be-Leased', '').replace('**', '').strip() if tenant_name_raw else None
        
        # Row 2: Suite ID
        row2 = str(group.iloc[1, 0]) if len(group) > 1 else ''
        suite_id = SmartParser.extract_pattern(row2, r'Suite:\s*([A-Z0-9-]+)')
        
        # Row 3: Date range
        row3 = str(group.iloc[2, 0]) if len(group) > 2 else ''
        lease_start = None
        lease_end = None
        if ' - ' in row3:
            parts = row3.split(' - ')
            lease_start = SmartParser.parse_date(parts[0])
            lease_end = SmartParser.parse_date(parts[1])
        
        # Row 4: Lease term
        lease_term = str(group.iloc[3, 0]).strip() if len(group) > 3 and pd.notna(group.iloc[3, 0]) else None
        if lease_term == 'nan':
            lease_term = None
        
        # Row 5: Tenure
        tenure = str(group.iloc[4, 0]).strip() if len(group) > 4 and pd.notna(group.iloc[4, 0]) else None
        if tenure == 'nan':
            tenure = None
        
        # Helper function to safely get cell value
        def get_val(row, col):
            try:
                if len(group) > row and len(group.iloc[row]) > col:
                    return group.iloc[row, col]
            except:
                pass
            return None
        
        # Area and building share
        initial_area = SmartParser.parse_number(get_val(0, 1))
        building_share = SmartParser.parse_number(get_val(1, 1))
        if building_share and building_share < 1:
            building_share *= 100
        
        # Lease status and space type
        lease_type_raw = str(get_val(1, 2))
        lease_status = 'Speculative' if 'Speculative' in lease_type_raw or is_speculative else 'Contract'
        
        space_raw = str(get_val(2, 2))
        space_category = None
        if 'OFFICE' in space_raw.upper():
            space_category = 'Office'
        elif 'RETAIL' in space_raw.upper():
            space_category = 'Retail'
        elif 'PARKING' in space_raw.upper():
            space_category = 'Parking'
        
        space_type = str(get_val(3, 2)).strip() if pd.notna(get_val(3, 2)) else None
        if space_type == 'nan':
            space_type = None
        
        # Rates and amounts
        base_rate_annual = SmartParser.parse_number(get_val(0, 3))
        base_amount_annual = SmartParser.parse_number(get_val(1, 3))
        base_rate_monthly = SmartParser.parse_number(get_val(2, 3))
        base_amount_monthly = SmartParser.parse_number(get_val(3, 3))
        rental_value_annual = SmartParser.parse_number(get_val(4, 3))
        
        # Free rent
        free_rent_date = SmartParser.parse_date(get_val(0, column_map.get('free_rent_date', 8)))
        free_rent_months_raw = str(get_val(0, column_map.get('free_rent_months', 9)))
        free_rent_months = None
        if free_rent_months_raw and free_rent_months_raw != 'nan':
            months_match = re.search(r'([\d.]+)\s*Months?', free_rent_months_raw, re.IGNORECASE)
            if months_match:
                free_rent_months = float(months_match.group(1))
        
        # Recovery structure
        recovery = str(get_val(0, column_map.get('recovery', 12)))
        if recovery in ['nan', '', 'None']:
            recovery = None
        
        # TI and Commission
        ti_col = column_map.get('ti', 14)
        ti_rate = SmartParser.parse_number(get_val(0, ti_col))
        ti_total = SmartParser.parse_number(get_val(1, ti_col))
        
        comm_col = column_map.get('commission', 17)
        commission_rate = SmartParser.parse_number(get_val(0, comm_col))
        commission_total = SmartParser.parse_number(get_val(1, comm_col))
        
        # Market renewal
        market_renewal_raw = str(get_val(0, column_map.get('market_renewal', 11)))
        market_renewal = None
        if market_renewal_raw and 'Market' in market_renewal_raw:
            pct_match = re.search(r'(\d+\.?\d*)\s*%', market_renewal_raw)
            if pct_match:
                market_renewal = float(pct_match.group(1))
        
        return {
            'suite_id': suite_id,
            'tenant_number': tenant_number,
            'tenant_name': tenant_name,
            'is_speculative': is_speculative,
            'lease_status': lease_status,
            'space_category': space_category,
            'space_type': space_type,
            'initial_area_sf': initial_area,
            'building_share_pct': building_share,
            'lease_start_date': lease_start,
            'lease_end_date': lease_end,
            'lease_term': lease_term,
            'tenure_type': tenure,
            'base_rate_annual': base_rate_annual,
            'base_rate_monthly': base_rate_monthly,
            'base_amount_annual': base_amount_annual,
            'base_amount_monthly': base_amount_monthly,
            'rental_value_annual': rental_value_annual,
            'free_rent_start_date': free_rent_date,
            'free_rent_months': free_rent_months,
            'recovery_structure': recovery,
            'ti_rate_per_sf': ti_rate,
            'ti_total_amount': ti_total,
            'commission_rate_per_sf': commission_rate,
            'commission_total_amount': commission_total,
            'market_renewal_pct': market_renewal
        }
    
    def _extract_rent_escalations(self, tenant_groups: List[pd.DataFrame], 
                                   column_map: Dict) -> pd.DataFrame:
        """Extract ALL rent escalations from all groups"""
        
        escalations = []
        
        for group in tenant_groups:
            try:
                # Get identifiers
                row2 = str(group.iloc[1, 0]) if len(group) > 1 else ''
                suite_id = SmartParser.extract_pattern(row2, r'Suite:\s*([A-Z0-9-]+)')
                
                row1 = str(group.iloc[0, 0]) if pd.notna(group.iloc[0, 0]) else ''
                tenant_match = re.match(r'^\d+\.\s+(.+)', row1)
                tenant_name = tenant_match.group(1).strip() if tenant_match else None
                if tenant_name:
                    tenant_name = tenant_name.replace('**To-Be-Leased', '').replace('**', '').strip()
                
                # Collect escalation data from ALL rows in the group
                all_dates = []
                all_rates_annual = []
                all_rates_monthly = []
                
                date_col = column_map.get('escalation_dates', 4)
                annual_col = column_map.get('escalation_annual', 5)
                monthly_col = column_map.get('escalation_monthly', 6)
                
                for row_idx in range(len(group)):
                    # Get cell values
                    def get_cell(col):
                        try:
                            if len(group.iloc[row_idx]) > col:
                                return group.iloc[row_idx, col]
                        except:
                            pass
                        return None
                    
                    dates_val = get_cell(date_col)
                    annual_val = get_cell(annual_col)
                    monthly_val = get_cell(monthly_col)
                    
                    # Split and accumulate
                    if dates_val:
                        all_dates.extend(SmartParser.split_delimited(dates_val))
                    if annual_val:
                        all_rates_annual.extend(SmartParser.split_delimited(annual_val))
                    if monthly_val:
                        all_rates_monthly.extend(SmartParser.split_delimited(monthly_val))
                
                # Create escalation records maintaining date-rate relationships
                max_len = max(len(all_dates), len(all_rates_annual), len(all_rates_monthly))
                
                for i in range(max_len):
                    escalation_date = SmartParser.parse_date(all_dates[i]) if i < len(all_dates) else None
                    rate_annual = SmartParser.parse_number(all_rates_annual[i]) if i < len(all_rates_annual) else None
                    rate_monthly = SmartParser.parse_number(all_rates_monthly[i]) if i < len(all_rates_monthly) else None
                    
                    if escalation_date and (rate_annual is not None or rate_monthly is not None):
                        escalations.append({
                            'suite_id': suite_id,
                            'tenant_name': tenant_name,
                            'escalation_date': escalation_date,
                            'new_rate_annual': rate_annual,
                            'new_rate_monthly': rate_monthly,
                            'escalation_sequence': i + 1
                        })
            
            except Exception as e:
                print(f"        Warning: Failed to extract escalations: {e}")
                continue
        
        df = pd.DataFrame(escalations)
        if len(df) > 0:
            df = df.sort_values(['suite_id', 'escalation_date'])
        
        return df
    
    def export_results(self):
        """Export all results to CSV files"""
        
        if not self.results:
            print("\n No data to export")
            return
        
        print(f"\n{'='*80}")
        print(" EXPORTING RESULTS")
        print('='*80)
        
        for filename, data in self.results.items():
            # Export lease master
            lease_path = self.output_dir / f'{filename}_lease_master.csv'
            data['lease_master'].to_csv(lease_path, index=False, encoding='utf-8')
            print(f" {lease_path.name} ({len(data['lease_master'])} records)")
            
            # Export escalations
            escalation_path = self.output_dir / f'{filename}_rent_escalations.csv'
            data['rent_escalations'].to_csv(escalation_path, index=False, encoding='utf-8')
            print(f" {escalation_path.name} ({len(data['rent_escalations'])} records)")
        
        print(f"\n All files saved to: {self.output_dir}/")
    
    def print_summary(self):
        """Print extraction summary"""
        
        print(f"\n{'='*80}")
        print(" EXTRACTION SUMMARY")
        print('='*80)
        
        for filename, data in self.results.items():
            lease_df = data['lease_master']
            escalation_df = data['rent_escalations']
            
            print(f"\n {filename}:")
            print(f"   Leases: {len(lease_df)}")
            print(f"   Escalations: {len(escalation_df)}")
            
            if len(escalation_df) > 0:
                by_suite = escalation_df.groupby('suite_id').size()
                print(f"   Suites with escalations: {len(by_suite)}")
                print(f"   Avg escalations per suite: {by_suite.mean():.1f}")
                print(f"   Max escalations: {by_suite.max()}")
        
        print(f"\n{'='*80}")
    
    def run_auto(self):
        """Fully automated execution"""
        
        # Auto-discover files
        files = self.auto_discover_files()
        
        if not files:
            print("\n No lease files found in current directory")
            print("\nLooking for files containing: 'ae', 'lease', 'rent', 'tenant', etc.")
            return False
        
        # Process all files
        success = self.process_files(files)
        
        if not success:
            return False
        
        # Export results
        self.export_results()
        
        # Print summary
        self.print_summary()
        
        print(f"\n{'='*80}")
        print(" "*25 + " EXTRACTION COMPLETE")
        print('='*80 + "\n")
        
        return True


def main():
    """Main entry point with flexible options"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Intelligent Lease Data Extractor - Zero Configuration Required',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and process all files
  python extractor.py
  
  # Process specific files
  python extractor.py --files "142 Prior AE.xls" "Owner AE.xls"
  
  # Specify output directory
  python extractor.py --output ./my_output
  
  # Process files from different directory
  python extractor.py --search-dir /path/to/files
        """
    )
    
    parser.add_argument(
        '--files',
        nargs='+',
        help='Specific files to process (skips auto-discovery)'
    )
    
    parser.add_argument(
        '--output',
        default='./extracted_data',
        help='Output directory for CSV files (default: ./extracted_data)'
    )
    
    parser.add_argument(
        '--search-dir',
        default='.',
        help='Directory to search for files (default: current directory)'
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Fully automatic mode (no prompts)'
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = IntelligentLeaseExtractor(output_dir=args.output)
    
    # Determine which files to process
    if args.files:
        # Use specified files
        files_to_process = args.files
        
        # Validate files exist
        missing = [f for f in files_to_process if not Path(f).exists()]
        if missing:
            print(f"\n Error: The following files were not found:")
            for f in missing:
                print(f"   • {f}")
            sys.exit(1)
        
        print(f"\n Processing {len(files_to_process)} specified file(s)")
        success = extractor.process_files(files_to_process)
    
    else:
        # Auto-discover files
        if args.auto:
            print(" Running in fully automatic mode...")
            success = extractor.run_auto()
        else:
            files = extractor.auto_discover_files(args.search_dir)
            
            if not files:
                print(f"\n No lease files found in '{args.search_dir}'")
                print("\nTip: Use --files to specify files manually")
                sys.exit(1)
            
            # Show discovered files and confirm
            print(f"\n Discovered {len(files)} file(s) in '{args.search_dir}':")
            for i, f in enumerate(files, 1):
                print(f"   {i}. {Path(f).name}")
            
            response = input("\nProcess these files? [Y/n]: ").strip().lower()
            
            if response in ['n', 'no']:
                print("Cancelled.")
                sys.exit(0)
            
            success = extractor.process_files(files)
    
    # Export and summarize if successful
    if success:
        extractor.export_results()
        extractor.print_summary()
        
        print(f"\n{'='*80}")
        print(" "*25 + "[OK] EXTRACTION COMPLETE")
        print('='*80 + "\n")
        
        sys.exit(0)
    else:
        print("\n Extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
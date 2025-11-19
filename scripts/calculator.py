import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("calculator")


class ImprovedAuditCalculator:
    """Enhanced calculator with detailed escalation output"""
    
    def __init__(self):
        self.thresholds = {
            'area_sf_abs': 100,
            'area_sf_pct': 5,
            'rent_abs': 1000,
            'rent_pct': 2,
            'rent_per_sf_abs': 2,
            'date_days': 7
        }
        
        self.findings = {
            'lease_status': [],
            'tenant_name': [],
            'lease_expiration': [],
            'leased_area': [],
            'base_rent': [],
            'rent_per_sf': []
        }
    
    def fuzzy_match_names(self, name1: str, name2: str, threshold: float = 0.7) -> bool:
        """Fuzzy match tenant names"""
        if pd.isna(name1) or pd.isna(name2):
            return False
        
        n1 = str(name1).strip().lower()
        n2 = str(name2).strip().lower()
        
        # Remove common suffixes
        for word in ['llc', 'inc', 'corp', 'ltd', 'company', 'co', 'corporation']:
            n1 = n1.replace(word, '').strip()
            n2 = n2.replace(word, '').strip()
        
        if n1 == n2:
            return True
        
        if n1 in n2 or n2 in n1:
            return True
        
        # Word overlap
        words1 = set(n1.split())
        words2 = set(n2.split())
        
        if words1 and words2:
            overlap = len(words1 & words2)
            min_len = min(len(words1), len(words2))
            if (overlap / min_len) >= threshold:
                return True
        
        return SequenceMatcher(None, n1, n2).ratio() >= threshold
    
    def calculate_variance(self, val1: float, val2: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate absolute and percentage variance"""
        if pd.isna(val1) or pd.isna(val2):
            return None, None
        
        try:
            v1, v2 = float(val1), float(val2)
            abs_var = v1 - v2
            pct_var = (abs_var / abs(v1)) * 100 if v1 != 0 else None
            return abs_var, pct_var
        except:
            return None, None
    
    def calculate_date_diff(self, date1: any, date2: any) -> Optional[int]:
        """Calculate days between dates"""
        if pd.isna(date1) or pd.isna(date2):
            return None
        
        try:
            d1 = pd.to_datetime(date1)
            d2 = pd.to_datetime(date2)
            return abs((d1 - d2).days)
        except:
            return None
    
    def merge_main_sources(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge 4 main sources on suite_number"""
        
        logger.info("\n" + "="*80)
        logger.info("MERGING MAIN SOURCES")
        logger.info("="*80)
        
        sources = {}
        
        for key, df in dfs.items():
            if 'prior_ae_lease_master' in key:
                sources['prior'] = df.add_suffix('_prior')
                logger.info(f"✓ Prior AE: {len(df)} records")
            elif 'owner_ae_lease_master' in key:
                sources['owner'] = df.add_suffix('_owner')
                logger.info(f"✓ Owner AE: {len(df)} records")
            elif 'rent_roll_suites' in key:
                sources['rentroll'] = df.add_suffix('_rentroll')
                logger.info(f"✓ Rent Roll: {len(df)} records")
            elif 'stacking_plan' in key:
                sources['stacking'] = df.add_suffix('_stacking')
                logger.info(f"✓ Stacking Plan: {len(df)} records")
        
        if len(sources) < 2:
            logger.error("Need at least 2 sources!")
            return pd.DataFrame()
        
        merged = None
        for source_name, df in sources.items():
            df_copy = df.copy()
            df_copy.rename(columns={f'suite_number_{source_name}': 'suite_number'}, inplace=True)
            
            if merged is None:
                merged = df_copy
            else:
                merged = merged.merge(df_copy, on='suite_number', how='outer')
        
        logger.info(f"✓ Merged: {len(merged)} unique suites")
        
        return merged
    
    def compare_lease_status(self, merged: pd.DataFrame) -> List[Dict]:
        """Section 1: Lease Status discrepancies"""
        
        logger.info("\n🔍 Comparing lease status...")
        
        findings = []
        
        for _, row in merged.iterrows():
            suite = row['suite_number']
            
            prior = row.get('lease_status_prior')
            owner = row.get('lease_status_owner')
            rentroll = row.get('lease_status_rentroll')
            stacking = row.get('lease_status_stacking')
            
            statuses = {
                'prior_nkf': prior,
                'owners_model': owner,
                'rent_roll': rentroll,
                'stacking_plan': stacking
            }
            
            # Filter out NaN
            statuses = {k: (v if pd.notna(v) else 'NAV') for k, v in statuses.items()}
            
            # Check if all same
            valid_statuses = [v for v in statuses.values() if v != 'NAV']
            if len(set(valid_statuses)) > 1 and len(valid_statuses) >= 2:
                findings.append({
                    'suite_number': suite,
                    'prior_nkf': statuses['prior_nkf'],
                    'owners_model': statuses['owners_model'],
                    'rent_roll': statuses['rent_roll'],
                    'stacking_plan': statuses['stacking_plan'],
                    'leases': 'NAV'
                })
        
        logger.info(f"  Found: {len(findings)} discrepancies")
        return findings
    
    def compare_tenant_names(self, merged: pd.DataFrame) -> List[Dict]:
        """Section 2: Tenant Name discrepancies"""
        
        logger.info("🔍 Comparing tenant names...")
        
        findings = []
        
        for _, row in merged.iterrows():
            suite = row['suite_number']
            
            prior = row.get('tenant_name_prior')
            owner = row.get('tenant_name_owner')
            rentroll = row.get('tenant_name_rentroll')
            stacking = row.get('tenant_name_stacking')
            
            # Skip if all vacant
            tenants = [t for t in [prior, owner, rentroll, stacking] 
                      if pd.notna(t) and str(t).lower() not in ['vacant', '']]
            
            if len(tenants) < 2:
                continue
            
            # Check for mismatches
            mismatched = False
            for i in range(len(tenants)):
                for j in range(i+1, len(tenants)):
                    if not self.fuzzy_match_names(tenants[i], tenants[j]):
                        mismatched = True
                        break
                if mismatched:
                    break
            
            if mismatched:
                findings.append({
                    'suite_number': suite,
                    'prior_nkf': prior if pd.notna(prior) else 'NAV',
                    'owners_model': owner if pd.notna(owner) else 'NAV',
                    'rent_roll': rentroll if pd.notna(rentroll) else 'NAV',
                    'stacking_plan': stacking if pd.notna(stacking) else 'NAV',
                    'leases': 'NAV'
                })
        
        logger.info(f"  Found: {len(findings)} discrepancies")
        return findings
    
    def compare_lease_expiration(self, merged: pd.DataFrame) -> List[Dict]:
        """Section 3: Lease Expiration Date discrepancies"""
        
        logger.info("🔍 Comparing lease expiration dates...")
        
        findings = []
        
        for _, row in merged.iterrows():
            suite = row['suite_number']
            
            tenant = (row.get('tenant_name_prior') or 
                     row.get('tenant_name_owner') or 
                     row.get('tenant_name_rentroll') or 
                     row.get('tenant_name_stacking'))
            
            prior = row.get('lease_end_date_prior')
            owner = row.get('lease_end_date_owner')
            rentroll = row.get('lease_end_date_rentroll')
            stacking = row.get('lease_end_date_stacking')
            
            dates = [d for d in [prior, owner, rentroll, stacking] if pd.notna(d)]
            
            if len(dates) < 2:
                continue
            
            # Check for mismatches
            unique_dates = set(dates)
            if len(unique_dates) > 1:
                findings.append({
                    'suite_number': suite,
                    'tenant_name': tenant if pd.notna(tenant) else '',
                    'prior_nkf': self._format_date(prior),
                    'owners_model': self._format_date(owner),
                    'rent_roll': self._format_date(rentroll),
                    'stacking_plan': self._format_date(stacking),
                    'leases': 'NAV'
                })
        
        logger.info(f"  Found: {len(findings)} discrepancies")
        return findings
    
    def compare_leased_area(self, merged: pd.DataFrame) -> List[Dict]:
        """Section 4: Leased Area with variances"""
        
        logger.info("🔍 Comparing leased area...")
        
        findings = []
        
        for _, row in merged.iterrows():
            suite = row['suite_number']
            
            tenant = (row.get('tenant_name_prior') or 
                     row.get('tenant_name_owner') or 
                     row.get('tenant_name_rentroll'))
            
            area_a = row.get('leased_area_sf_prior')
            area_b = row.get('leased_area_sf_owner')
            area_c = row.get('leased_area_sf_rentroll')
            area_d = row.get('leased_area_sf_stacking')
            
            var_ab, _ = self.calculate_variance(area_a, area_b)
            var_ac, _ = self.calculate_variance(area_a, area_c)
            var_ad, _ = self.calculate_variance(area_a, area_d)
            
            has_variance = False
            for var in [var_ab, var_ac, var_ad]:
                if var is not None and abs(var) > self.thresholds['area_sf_abs']:
                    has_variance = True
                    break
            
            if has_variance:
                findings.append({
                    'suite_number': suite,
                    'tenant_name': tenant if pd.notna(tenant) else '',
                    'prior_nkf_a': self._format_number(area_a),
                    'owners_model_b': self._format_number(area_b),
                    'rent_roll_c': self._format_number(area_c),
                    'stacking_plan_d': self._format_number(area_d),
                    'variance_a_b': self._format_number(var_ab),
                    'variance_a_c': self._format_number(var_ac),
                    'variance_a_d': self._format_number(var_ad)
                })
        
        logger.info(f"  Found: {len(findings)} discrepancies")
        return findings
    
    def compare_base_rent(self, merged: pd.DataFrame, escalation_dfs: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Section 5: Base Rent with DETAILED escalation schedules"""
        
        logger.info("🔍 Comparing base rent & escalations...")
        
        findings = []
        
        for _, row in merged.iterrows():
            suite = row['suite_number']
            
            tenant = (row.get('tenant_name_prior') or 
                     row.get('tenant_name_owner') or 
                     row.get('tenant_name_rentroll'))
            
            # Get base rents and areas
            rent_prior = row.get('amount_per_year_prior')
            rent_owner = row.get('amount_per_year_owner')
            rent_rentroll = row.get('amount_per_year_rentroll')
            
            area_prior = row.get('leased_area_sf_prior')
            area_owner = row.get('leased_area_sf_owner')
            area_rentroll = row.get('leased_area_sf_rentroll')
            
            # Check for base rent variance
            var_ab, _ = self.calculate_variance(rent_prior, rent_owner)
            var_ac, _ = self.calculate_variance(rent_prior, rent_rentroll)
            
            has_rent_variance = False
            for var in [var_ab, var_ac]:
                if var is not None and abs(var) > self.thresholds['rent_abs']:
                    has_rent_variance = True
                    break
            
            # Get escalation schedules
            prior_escalations = self._get_escalations_for_suite(suite, escalation_dfs, 'prior_ae', area_prior)
            owner_escalations = self._get_escalations_for_suite(suite, escalation_dfs, 'owner_ae', area_owner)
            rentroll_escalations = self._get_escalations_for_suite(suite, escalation_dfs, 'rent_roll', area_rentroll)
            
            # Check for escalation count mismatch
            esc_counts = [len(prior_escalations), len(owner_escalations), len(rentroll_escalations)]
            has_esc_mismatch = len(set([c for c in esc_counts if c > 0])) > 1
            
            if has_rent_variance or has_esc_mismatch:
                finding = {
                    'suite_number': suite,
                    'tenant_name': tenant if pd.notna(tenant) else '',
                    'prior_nkf': self._format_escalation_schedule(rent_prior, prior_escalations),
                    'owners_model': self._format_escalation_schedule(rent_owner, owner_escalations),
                    'rent_roll': self._format_escalation_schedule(rent_rentroll, rentroll_escalations),
                    'leases': 'NAV'
                }
                findings.append(finding)
        
        logger.info(f"  Found: {len(findings)} discrepancies")
        return findings
    
    def _get_escalations_for_suite(self, suite: str, escalation_dfs: Dict, source_prefix: str, area_sf: float) -> List[Dict]:
        """Get escalation schedule for a suite from a specific source"""
        
        escalations = []
        
        for key, esc_df in escalation_dfs.items():
            if esc_df is None or len(esc_df) == 0:
                continue
            
            # Match source - be flexible with naming
            if source_prefix == 'rent_roll':
                if 'rent_roll' not in key and 'increase' not in key:
                    continue
            elif source_prefix not in key:
                continue
            
            suite_escs = esc_df[esc_df['suite_number'] == suite].copy()
            
            if len(suite_escs) == 0:
                continue
            
            # Determine date column
            date_col = None
            if 'escalation_date' in suite_escs.columns:
                date_col = 'escalation_date'
            elif 'increase_date' in suite_escs.columns:
                date_col = 'increase_date'
            else:
                continue
            
            # Sort by date
            suite_escs = suite_escs.sort_values(date_col)
            
            for _, esc_row in suite_escs.iterrows():
                date_val = esc_row.get(date_col)
                
                if pd.isna(date_val):
                    continue
                
                # Get annual amount - try multiple methods
                annual_amount = None
                
                # Method 1: rate_annual * area (for Prior/Owner AE)
                if 'rate_annual' in esc_row.index and pd.notna(esc_row['rate_annual']):
                    rate = float(esc_row['rate_annual'])
                    if pd.notna(area_sf) and area_sf > 0:
                        annual_amount = rate * area_sf
                
                # Method 2: monthly_amount * 12 (for Rent Roll)
                elif 'monthly_amount' in esc_row.index and pd.notna(esc_row['monthly_amount']):
                    monthly = float(esc_row['monthly_amount'])
                    annual_amount = monthly * 12
                
                # Method 3: psf * area (alternative for Rent Roll)
                elif 'psf' in esc_row.index and pd.notna(esc_row['psf']):
                    psf = float(esc_row['psf'])
                    if pd.notna(area_sf) and area_sf > 0:
                        annual_amount = psf * area_sf
                
                # Add to list if we have a valid amount
                if pd.notna(annual_amount) and annual_amount > 0:
                    escalations.append({
                        'date': date_val,
                        'amount_per_year': annual_amount
                    })
        
        return escalations
    
    def _format_escalation_schedule(self, base_rent: float, escalations: List[Dict]) -> str:
        """Format escalation schedule as JSON string for CSV storage"""
        
        schedule = []
        
        # Add base rent
        if pd.notna(base_rent):
            schedule.append({
                'period': 'Base Rent',
                'date': '',
                'amount_per_year': f"${float(base_rent):,.2f}"
            })
        
        # Add escalations
        for esc in escalations:
            date_str = self._format_date(esc['date'])
            amount = esc.get('amount_per_year')
            
            schedule.append({
                'period': 'Escalation',
                'date': date_str,
                'amount_per_year': f"${float(amount):,.2f}" if pd.notna(amount) else ''
            })
        
        return json.dumps(schedule)
    
    def compare_rent_per_sf(self, merged: pd.DataFrame) -> List[Dict]:
        """Section 6: Rent Per SF discrepancies"""
        
        logger.info("🔍 Comparing rent per SF...")
        
        findings = []
        
        for _, row in merged.iterrows():
            suite = row['suite_number']
            
            tenant = (row.get('tenant_name_prior') or 
                     row.get('tenant_name_owner') or 
                     row.get('tenant_name_rentroll'))
            
            rate_prior = row.get('rent_per_sf_prior')
            rate_owner = row.get('rent_per_sf_owner')
            rate_rentroll = row.get('rent_per_sf_rentroll')
            
            var_ab, _ = self.calculate_variance(rate_prior, rate_owner)
            var_ac, _ = self.calculate_variance(rate_prior, rate_rentroll)
            
            has_variance = False
            for var in [var_ab, var_ac]:
                if var is not None and abs(var) > self.thresholds['rent_per_sf_abs']:
                    has_variance = True
                    break
            
            if has_variance:
                findings.append({
                    'suite_number': suite,
                    'tenant_name': tenant if pd.notna(tenant) else '',
                    'prior_rate': self._format_currency(rate_prior, is_rate=True),
                    'owner_rate': self._format_currency(rate_owner, is_rate=True),
                    'rentroll_rate': self._format_currency(rate_rentroll, is_rate=True),
                    'variance_a_b': self._format_currency(var_ab, is_rate=True),
                    'variance_a_c': self._format_currency(var_ac, is_rate=True)
                })
        
        logger.info(f"  Found: {len(findings)} discrepancies")
        return findings
    
    def _format_date(self, date_val):
        """Format date for display"""
        if pd.isna(date_val):
            return 'NAV'
        try:
            return pd.to_datetime(date_val).strftime('%m/%d/%y')
        except:
            return str(date_val)
    
    def _format_number(self, num_val):
        """Format number with commas"""
        if pd.isna(num_val):
            return ''
        try:
            return f"{float(num_val):,.0f}"
        except:
            return ''
    
    def _format_currency(self, num_val, is_rate=False):
        """Format currency"""
        if pd.isna(num_val):
            return ''
        try:
            if is_rate:
                return f"${float(num_val):.2f}"
            else:
                return f"${float(num_val):,.2f}"
        except:
            return ''
    
    def run_all_comparisons(self, validated_dfs: Dict[str, pd.DataFrame]) -> Dict:
        """Run all audit comparisons"""
        
        logger.info("\n" + "="*80)
        logger.info("RUNNING IMPROVED AUDIT CALCULATIONS")
        logger.info("="*80)
        
        # Separate lease masters from escalations
        lease_masters = {k: v for k, v in validated_dfs.items() 
                        if 'lease_master' in k or 'suites' in k or 'stacking' in k}
        
        escalations = {k: v for k, v in validated_dfs.items() 
                      if 'escalation' in k or 'increase' in k}
        
        # Merge main sources
        merged = self.merge_main_sources(lease_masters)
        
        if merged.empty:
            logger.error("No data to compare!")
            return self.findings
        
        # Run all comparisons
        self.findings['lease_status'] = self.compare_lease_status(merged)
        self.findings['tenant_name'] = self.compare_tenant_names(merged)
        self.findings['lease_expiration'] = self.compare_lease_expiration(merged)
        self.findings['leased_area'] = self.compare_leased_area(merged)
        self.findings['base_rent'] = self.compare_base_rent(merged, escalations)
        self.findings['rent_per_sf'] = self.compare_rent_per_sf(merged)
        
        # Summary
        total = sum(len(f) for f in self.findings.values())
        
        logger.info("\n" + "="*80)
        logger.info(f"AUDIT COMPLETE - Total Discrepancies: {total}")
        for category, findings in self.findings.items():
            if findings:
                logger.info(f"  • {category.replace('_', ' ').title():30s} {len(findings):>3}")
        logger.info("="*80)
        
        return self.findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with validated CSVs")
    parser.add_argument("--output", required=True, help="Output directory for findings")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load all validated CSVs
    validated_dfs = {}
    for csv_file in input_dir.glob("*_validated.csv"):
        key = csv_file.stem.replace('_validated', '')
        validated_dfs[key] = pd.read_csv(csv_file)
        logger.info(f"Loaded: {key}")
    
    # Run calculations
    calculator = ImprovedAuditCalculator()
    findings = calculator.run_all_comparisons(validated_dfs)
    
    # Save findings
    for category, finding_list in findings.items():
        if finding_list:
            df = pd.DataFrame(finding_list)
            df.to_csv(output_dir / f"{category}_findings.csv", index=False)
            logger.info(f"Saved: {category}_findings.csv")
    
    logger.info("\n All calculations complete!")


if __name__ == "__main__":
    main()
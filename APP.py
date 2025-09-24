import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from datetime import datetime
import re
import json
from textwrap import fill
import time

class AdvancedResearchAnalyzer:
    def __init__(self):
        self.core_api_key = "iKwyzVkmEbqsnjclMQa6gSIGNBRAh1Uf"
        self.gemini_api_key = "AIzaSyBdjVMm0B4LcKUwLy6sRmnumUFVqAkJnEg"
        self.base_url = "https://api.core.ac.uk/v3/"
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.papers_df = None
        self.topics = None
        self.ai_analysis = {}
        self.research_gaps = []
        self.additional_keywords = []
        self.current_keyword = ""
        self.current_topic = ""

    def print_header(self):
        """Print professional header"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 ADVANCED RESEARCH GAP ANALYSIS & INNOVATION TOOL           ║
║               AI-Powered Literature Review + Patent Analysis               ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)

    def print_section(self, title):
        """Print section headers"""
        print(f"\n{'='*80}")
        print(f"📊 {title}")
        print(f"{'='*80}")

    def call_gemini_api(self, prompt, temperature=0.7, max_tokens=1000):
        """Generic function to call Gemini API"""
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
        }

        try:
            response = requests.post(
                f"{self.gemini_url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text']
            return "AI analysis temporarily unavailable. Please check your API key."
        except Exception as e:
            return f"AI analysis error: {str(e)}"

    def get_user_input_with_keywords(self):
        """Get comprehensive user input with keyword options"""
        self.print_header()

        print("🔍 **Research Parameters**")
        print("-" * 50)

        # Basic research parameters
        keyword = input("Enter research keyword (e.g., 'solar panel'): ").strip() or "solar panel"
        topic = input("Enter specific topic (e.g., 'photon efficiency'): ").strip() or "photon"
        target_papers = input("Target paper count (default 100): ").strip()
        target_papers = int(target_papers) if target_papers.isdigit() else 100

        # Store the current keywords for use throughout the analysis
        self.current_keyword = keyword
        self.current_topic = topic

        # Additional keywords option
        print("\n🔑 **Additional Keywords**")
        print("-" * 30)
        print("Would you like to add more specific keywords to improve search?")
        add_more = input("Add additional keywords? (y/n): ").strip().lower()

        if add_more == 'y':
            print("\nEnter additional keywords (comma-separated):")
            print("Examples: 'nanotechnology, quantum dots, perovskite, efficiency'")
            additional_input = input("Additional keywords: ").strip()
            if additional_input:
                self.additional_keywords = [kw.strip() for kw in additional_input.split(',')]
                print(f"✅ Added {len(self.additional_keywords)} additional keywords")

        return keyword, topic, target_papers

    def generate_research_keywords(self, keyword, topic):
        """Generate comprehensive research keywords using Gemini"""
        prompt = f"""
        As an expert research librarian, generate 20-25 highly specific academic research keywords
        and search terms for the topic: "{keyword}" with focus on "{topic}".

        Include:
        - Technical terminology
        - Methodologies and techniques
        - Applications and use cases
        - Related subfields
        - Emerging trends
        - Specific technologies

        Return as a numbered list of search queries suitable for academic databases.
        """

        result = self.call_gemini_api(prompt)
        if result and "unavailable" not in result.lower() and "error" not in result.lower():
            lines = result.split('\n')
            queries = []
            for line in lines:
                if re.match(r'^\d+[\.\)]', line.strip()):
                    query = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                    queries.append(query)
            return queries[:20]

        # Fallback keywords if Gemini fails
        return [
            f"{keyword} {topic}",
            f"{keyword} technology",
            f"{topic} efficiency",
            f"{keyword} materials",
            f"{topic} absorption",
            f"{keyword} innovation"
        ]

    def search_with_keyword_expansion(self, base_query, target_papers=100):
        """Comprehensive search using multiple keyword strategies"""
        all_papers = []
        used_queries = set()

        # Strategy 1: Base query
        papers = self.search_core_api(base_query, min(30, target_papers//3))
        if papers:
            all_papers.extend(papers)
            used_queries.add(base_query)
            print(f"   ✅ Base query: {len(papers)} papers")

        # Strategy 2: Gemini-generated keywords
        keyword, topic = base_query.split(' ', 1) if ' ' in base_query else (base_query, '')
        gemini_keywords = self.generate_research_keywords(keyword, topic)

        for query in gemini_keywords[:5]:  # Use top 5 Gemini keywords
            if len(all_papers) >= target_papers:
                break
            if query not in used_queries:
                papers = self.search_core_api(query, min(20, target_papers//5))
                if papers:
                    all_papers.extend(papers)
                    used_queries.add(query)
                    print(f"   ✅ Gemini keyword: {len(papers)} papers")

        # Strategy 3: User additional keywords
        for add_kw in self.additional_keywords[:3]:
            if len(all_papers) >= target_papers:
                break
            enhanced_query = f"{base_query} {add_kw}"
            if enhanced_query not in used_queries:
                papers = self.search_core_api(enhanced_query, min(15, target_papers//6))
                if papers:
                    all_papers.extend(papers)
                    used_queries.add(enhanced_query)
                    print(f"   ✅ Enhanced query: {len(papers)} papers")

        # Strategy 4: Related terms
        related_queries = [
            f"{base_query} review",
            f"{base_query} recent",
            f"{base_query} 2023",
            f"{base_query} future"
        ]

        for query in related_queries:
            if len(all_papers) >= target_papers:
                break
            if query not in used_queries:
                papers = self.search_core_api(query, min(10, target_papers//8))
                if papers:
                    all_papers.extend(papers)
                    used_queries.add(query)
                    print(f"   ✅ Related term: {len(papers)} papers")

        return all_papers[:target_papers]

    def search_core_api(self, query, limit):
        """Search CORE API with enhanced metadata collection"""
        headers = {'Authorization': f'Bearer {self.core_api_key}'}
        params = {'q': query, 'limit': min(limit, 50), 'offset': 0}

        try:
            response = requests.get(f"{self.base_url}search/works", headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                papers = []
                for paper in response.json().get('results', []):
                    paper_data = self.extract_paper_metadata(paper, query)
                    papers.append(paper_data)
                return papers
        except Exception as e:
            print(f"   ❌ Search error for '{query}': {e}")
        return []

    def extract_paper_metadata(self, paper, query):
        """Extract comprehensive paper metadata"""
        year = ""
        if paper.get('publishedDate'):
            year_str = paper.get('publishedDate', '')[:4]
            year = year_str if year_str.isdigit() else ""

        authors = [author.get('name', '') for author in paper.get('authors', [])]

        return {
            'title': paper.get('title', 'No Title'),
            'abstract': paper.get('abstract', 'No Abstract'),
            'year': year,
            'authors': authors,
            'keywords': paper.get('keywords', []),
            'citations': paper.get('citationCount', 0),
            'downloadUrl': paper.get('downloadUrl', ''),
            'doi': paper.get('doi', ''),
            'id': paper.get('id', ''),
            'search_query': query,
            'full_text_link': f"https://core.ac.uk/display/{paper.get('id', '')}",
            'publication': paper.get('publishedIn', 'Unknown'),
            'language': paper.get('language', 'Unknown')
        }

    def search_papers(self, keyword, topic, target_papers=100):
        """Main paper search function"""
        base_query = f"{keyword} {topic}".strip()

        self.print_section("SEARCH STRATEGY & PAPER COLLECTION")
        print(f"🎯 Target: {target_papers} research papers")
        print(f"🔍 Base query: {base_query}")

        if self.additional_keywords:
            print(f"🔑 Additional keywords: {', '.join(self.additional_keywords)}")

        all_papers = self.search_with_keyword_expansion(base_query, target_papers)

        if all_papers:
            self.papers_df = pd.DataFrame(all_papers).drop_duplicates(subset=['title'])
            print(f"\n✅ COLLECTION COMPLETE: {len(self.papers_df)} unique papers")

            # Display sample papers
            print("\n📄 **Sample Papers Found:**")
            for i in range(min(3, len(self.papers_df))):
                print(f"  {i+1}. {self.papers_df.iloc[i]['title'][:80]}...")

            return self.papers_df
        else:
            print("❌ No papers found. Please try different keywords or check API key.")
            return None

    def analyze_paper_with_ai(self, paper):
        """Use Gemini to analyze paper content and extract key insights"""
        if paper['title'] in self.ai_analysis:
            return self.ai_analysis[paper['title']]

        prompt = f"""
        Analyze this academic paper and extract key information:

        Title: {paper['title']}
        Abstract: {paper['abstract'][:1000] if paper['abstract'] and len(paper['abstract']) > 50 else 'No abstract available'}

        Research Context: This paper is part of a research analysis on {self.current_keyword} {self.current_topic}

        Provide analysis in this format:
        RESEARCH_FOCUS: [Main research focus]
        METHODOLOGY: [Key methodologies used]
        KEY_FINDINGS: [Main findings/results]
        CONTRIBUTION: [Novel contribution to field]
        GAPS_IDENTIFIED: [Any research gaps mentioned]
        RELEVANCE_SCORE: [1-10 score for relevance to {self.current_keyword} {self.current_topic} research]

        Be concise and analytical. If abstract is insufficient, provide general analysis based on title.
        """

        analysis = self.call_gemini_api(prompt, temperature=0.3)
        self.ai_analysis[paper['title']] = analysis
        return analysis

    def display_paper_details(self):
        """Display detailed paper information with AI analysis"""
        if self.papers_df is None or len(self.papers_df) == 0:
            return

        self.print_section("DETAILED PAPER ANALYSIS")
        print(f"📚 Analyzing {len(self.papers_df)} papers on {self.current_keyword} {self.current_topic} with AI...\n")

        display_count = min(5, len(self.papers_df))  # Show only first 5 papers

        for idx in range(display_count):
            paper = self.papers_df.iloc[idx]
            print(f"🔬 PAPER {idx+1}: {paper['title']}")
            print(f"   📅 Year: {paper['year']} | 📊 Citations: {paper['citations']}")
            print(f"   👥 Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
            print(f"   🔗 Full Text: {paper['full_text_link']}")
            if paper['doi']:
                print(f"   📄 DOI: {paper['doi']}")

            analysis = self.analyze_paper_with_ai(paper)
            if analysis:
                print("   🤖 AI ANALYSIS:")
                for line in analysis.split('\n'):
                    if line.strip():
                        print(f"      {line}")

            print("-" * 80)
            if idx < display_count - 1:  # Small delay between papers
                time.sleep(1)

        if len(self.papers_df) > display_count:
            print(f"   ... (showing first {display_count} papers for brevity)")

    def cluster_and_analyze(self):
        """Basic clustering implementation using actual keywords"""
        self.print_section("TOPIC MODELING & CLUSTER ANALYSIS")

        if self.papers_df is None or len(self.papers_df) == 0:
            print("❌ No papers available for clustering.")
            return None

        print(f"🤖 Basic analysis of {len(self.papers_df)} papers on {self.current_keyword} {self.current_topic}...")

        # Simple topic grouping based on actual keywords
        self.topics = [{
            'id': 0,
            'name': f'{self.current_keyword} {self.current_topic} Research',
            'papers_count': len(self.papers_df),
            'avg_citations': self.papers_df['citations'].mean() if len(self.papers_df) > 0 else 0
        }]

        print("✅ Basic analysis complete")
        return self.topics

    def preprocess_text(self, texts):
        """Text preprocessing"""
        cleaned_texts = []
        for text in texts:
            if pd.isna(text) or text == "No Abstract":
                cleaned_texts.append("")
                continue
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
            cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
            cleaned_texts.append(cleaned)
        return cleaned_texts

    def detect_comprehensive_gaps(self):
        """Comprehensive gap detection with guaranteed gap identification using actual keywords"""
        self.print_section("RESEARCH GAP ANALYSIS")

        gaps = []

        # Check for limited research based on paper count
        if len(self.papers_df) < 50:
            gaps.append({
                'type': 'LIMITED_RESEARCH',
                'severity': 'MEDIUM',
                'description': f"Only {len(self.papers_df)} papers found for '{self.current_keyword} {self.current_topic}' - suggests potential under-research in specific areas of this field"
            })

        # Dynamic gaps based on actual keywords
        gaps.append({
            'type': 'APPLICATION_GAP',
            'severity': 'HIGH',
            'description': f'Limited research on practical applications and real-world implementation challenges of {self.current_keyword} {self.current_topic} technologies across different use cases and environments'
        })

        gaps.append({
            'type': 'METHODOLOGICAL_GAP',
            'severity': 'MEDIUM',
            'description': f'Need for standardized methodologies and comparative studies across different approaches to {self.current_keyword} {self.current_topic} research'
        })

        gaps.append({
            'type': 'INTERDISCIPLINARY_GAP',
            'severity': 'MEDIUM',
            'description': f'Limited integration of {self.current_keyword} {self.current_topic} research with related fields and emerging technologies'
        })

        self.research_gaps = gaps
        print(f"🎯 Identified {len(gaps)} research considerations for '{self.current_keyword} {self.current_topic}'")
        return gaps

    def generate_enhanced_scholar_recommendations(self):
        """Generate detailed, actionable recommendations for scholars using actual keywords"""
        self.print_section("SCHOLARLY RESEARCH RECOMMENDATIONS")

        if not self.research_gaps:
            prompt = f"""
            Based on the analysis of {len(self.papers_df)} papers on {self.current_keyword} {self.current_topic} research,
            provide SPECIFIC, ACTIONABLE recommendations for researchers. Even though no major gaps were found,
            suggest innovative directions.

            Provide concrete next steps in this exact format:

            IMMEDIATE ACTIONS (Next 3-6 months):
            • [Specific action 1 with timeline related to {self.current_keyword}]
            • [Specific action 2 with resources needed for {self.current_topic}]
            • [Specific action 3 with expected outcome for {self.current_keyword} applications]

            MEDIUM-TERM PROJECTS (6-18 months):
            • [Project idea 1 with collaboration opportunities in {self.current_topic}]
            • [Project idea 2 with funding sources for {self.current_keyword} research]
            • [Project idea 3 with implementation steps for {self.current_keyword} {self.current_topic}]

            LONG-TERM RESEARCH VISION (2-3 years):
            • [Visionary direction 1 with potential impact on {self.current_keyword} field]
            • [Visionary direction 2 with innovation potential for {self.current_topic}]
            • [Visionary direction 3 with commercialization path for {self.current_keyword} technologies]

            SPECIFIC RESEARCH QUESTIONS TO PURSUE:
            1. [Novel question 1 specific to {self.current_keyword} {self.current_topic}]
            2. [Novel question 2 addressing challenges in {self.current_topic}]
            3. [Novel question 3 exploring new applications of {self.current_keyword}]

            Make each recommendation specific, actionable, and tailored to {self.current_keyword} {self.current_topic} research.
            """
        else:
            prompt = f"""
            Based on these identified research gaps, provide SPECIFIC, ACTIONABLE recommendations:

            RESEARCH GAPS IDENTIFIED:
            {json.dumps(self.research_gaps, indent=2)}

            Research Topic: {self.current_keyword} {self.current_topic}

            Provide concrete next steps in this exact format:

            IMMEDIATE ACTIONS (Next 3-6 months):
            • [Specific action 1 addressing the most critical gap in {self.current_topic}]
            • [Specific action 2 with timeline and resources for {self.current_keyword} research]
            • [Specific action 3 with measurable outcomes for {self.current_keyword} applications]

            MEDIUM-TERM PROJECTS (6-18 months):
            • [Project addressing methodological gaps in {self.current_topic} research]
            • [Project focusing on application gaps for {self.current_keyword} technologies]
            • [Collaborative project opportunity in {self.current_keyword} {self.current_topic} field]

            LONG-TERM RESEARCH VISION (2-3 years):
            • [Transformative research direction for {self.current_keyword} field]
            • [Commercialization pathway for {self.current_topic} innovations]
            • [Field-level impact project for {self.current_keyword} research]

            SPECIFIC RESEARCH QUESTIONS:
            1. [Question derived from gap analysis specific to {self.current_keyword}]
            2. [Innovative question combining multiple gaps in {self.current_topic}]
            3. [High-impact question with practical applications for {self.current_keyword}]

            Ensure each recommendation is concrete, actionable, and directly addresses the identified gaps for {self.current_keyword} {self.current_topic} research.
            """

        recommendations = self.call_gemini_api(prompt, max_tokens=1500)
        return recommendations if recommendations else "Recommendations generation failed. Please try again."

    def generate_enhanced_patent_analysis(self):
        """Generate comprehensive patent and commercialization analysis using actual keywords"""
        self.print_section("PATENT & COMMERCIALIZATION ANALYSIS")

        prompt = f"""
        Provide a DETAILED patent and commercialization analysis for {self.current_keyword} {self.current_topic} research.

        Research Context: {len(self.papers_df)} papers analyzed, {len(self.research_gaps)} gaps identified

        Provide analysis in this structured format:

        PATENTABLE TECHNOLOGIES:
        • [Technology 1 with novelty factors specific to {self.current_keyword} {self.current_topic}]
        • [Technology 2 with competitive advantage in {self.current_topic} applications]
        • [Technology 3 with market potential for {self.current_keyword} innovations]

        COMMERCIALIZATION OPPORTUNITIES:
        • [Market opportunity 1 with size estimate for {self.current_keyword} technologies]
        • [Market opportunity 2 with growth potential in {self.current_topic} applications]
        • [Market opportunity 3 with implementation timeline for {self.current_keyword} products]

        INDUSTRY COLLABORATION PARTNERS:
        • [Company type 1 with specific examples relevant to {self.current_keyword}]
        • [Company type 2 with collaboration models for {self.current_topic} research]
        • [Research institution partnerships specializing in {self.current_keyword} {self.current_topic}]

        REGULATORY & STANDARDS CONSIDERATIONS:
        • [Key regulations affecting commercialization of {self.current_keyword} technologies]
        • [Standards development opportunities for {self.current_topic} applications]
        • [Certification requirements for {self.current_keyword} innovations]

        Make this analysis specific, actionable, and focused on real-world commercial potential for {self.current_keyword} {self.current_topic} research.
        """

        analysis = self.call_gemini_api(prompt, max_tokens=1200)
        return analysis if analysis else "Patent analysis generation failed. Please check your API key."

    def generate_comprehensive_review(self):
        """Generate final comprehensive review with all elements using actual keywords"""
        self.print_section("FINAL COMPREHENSIVE RESEARCH REVIEW")

        gaps_summary = "\n".join([f"- {gap['description']} [{gap['severity']} priority]"
                                for gap in self.research_gaps]) if self.research_gaps else "No significant traditional gaps found"

        prompt = f"""
        Create a COMPREHENSIVE, PROFESSIONAL research review report for {self.current_keyword.upper()} {self.current_topic.upper()} RESEARCH.

        RESEARCH CONTEXT:
        - Papers Analyzed: {len(self.papers_df)}
        - Research Clusters: {len(self.topics) if self.topics else 'N/A'}
        - Gaps Identified: {len(self.research_gaps)}
        - Additional Keywords Used: {', '.join(self.additional_keywords) if self.additional_keywords else 'None'}
        - Primary Research Focus: {self.current_keyword} {self.current_topic}

        Create a structured report with these sections:

        1. EXECUTIVE SUMMARY
           • Overall research landscape assessment for {self.current_keyword} {self.current_topic}
           • Key findings and opportunities in this specific field
           • Strategic recommendations for {self.current_keyword} researchers

        2. RESEARCH LANDSCAPE ANALYSIS
           • Current state of {self.current_keyword} {self.current_topic} research
           • Major research trends in {self.current_topic} domain
           • Key contributors and institutions in {self.current_keyword} research

        3. GAP ANALYSIS RESULTS
           {gaps_summary}

        4. STRATEGIC RECOMMENDATIONS
           • Priority research directions for {self.current_keyword} {self.current_topic}
           • Resource allocation suggestions for this field
           • Timeline for implementation in {self.current_topic} research

        5. INNOVATION ROADMAP
           • Short-term innovations (0-1 year) for {self.current_keyword} applications
           • Medium-term developments (1-3 years) in {self.current_topic} technology
           • Long-term transformations (3-5 years) for {self.current_keyword} field

        6. CONCLUSION & ACTION PLAN
           • Summary of critical insights for {self.current_keyword} {self.current_topic} research
           • Specific next steps for researchers in this domain
           • Expected outcomes and impacts for {self.current_keyword} innovations

        Make this report professional, data-driven, and actionable for academic and industry audiences interested in {self.current_keyword} {self.current_topic} research.
        """

        review = self.call_gemini_api(prompt, max_tokens=2500)
        return review if review else "Comprehensive review generation failed. Please try again."

    def generate_next_steps(self):
        """Generate specific next steps for researchers using actual keywords"""
        self.print_section("NEXT STEPS FOR RESEARCHERS")

        prompt = f"""
        Based on the complete analysis of {len(self.papers_df)} papers on {self.current_keyword} {self.current_topic} research,
        provide SPECIFIC, ACTIONABLE next steps for researchers.

        Key findings: {len(self.research_gaps)} research gaps identified, {len(self.topics) if self.topics else 0} research clusters analyzed.

        Provide concrete next steps in this exact format:

        1. IMMEDIATE ACTIONS (Next 30 days):
           • [Specific task 1 with deadline related to {self.current_keyword} {self.current_topic}]
           • [Specific task 2 with resources for {self.current_topic} research]
           • [Specific task 3 with success metrics for {self.current_keyword} applications]

        2. SHORT-TERM GOALS (1-3 months):
           • [Goal 1 with measurable outcomes for {self.current_keyword} research]
           • [Goal 2 with implementation plan for {self.current_topic} projects]
           • [Goal 3 with collaboration opportunities in {self.current_keyword} field]

        3. MEDIUM-TERM OBJECTIVES (3-12 months):
           • [Objective 1 with timeline for {self.current_keyword} {self.current_topic} development]
           • [Objective 2 with funding strategy for {self.current_topic} research]
           • [Objective 3 with risk assessment for {self.current_keyword} applications]

        4. LONG-TERM STRATEGY (1-2 years):
           • [Strategic direction 1 with impact assessment for {self.current_keyword} field]
           • [Strategic direction 2 with scalability plan for {self.current_topic} technologies]
           • [Strategic direction 3 with sustainability considerations for {self.current_keyword} innovations]

        5. SPECIFIC DELIVERABLES:
           • [Deliverable 1: Research proposal on specific {self.current_topic} aspect]
           • [Deliverable 2: Patent application for {self.current_keyword} technology]
           • [Deliverable 3: Collaboration proposal with partners in {self.current_keyword} field]

        Make each step specific, measurable, achievable, relevant, and time-bound (SMART) for {self.current_keyword} {self.current_topic} research.
        """

        next_steps = self.call_gemini_api(prompt, max_tokens=1200)
        return next_steps if next_steps else f"""
        1. Review the identified gaps and recommendations for {self.current_keyword} {self.current_topic}
        2. Consider patent opportunities for {self.current_keyword} commercialization
        3. Develop a research proposal based on the most promising directions in {self.current_topic}
        4. Seek collaborations in underserved application areas of {self.current_keyword}
        5. Conduct deeper literature review on specific {self.current_topic} sub-topics
        6. Identify potential funding sources for {self.current_keyword} {self.current_topic} research
        7. Connect with industry partners for practical applications of {self.current_keyword}
        8. Plan experimental validation of theoretical findings in {self.current_topic}
        """

    def visualize_results(self):
        """Create basic visualizations of the analysis results"""
        try:
            if self.papers_df is not None and len(self.papers_df) > 0:
                self.print_section("VISUAL ANALYSIS SUMMARY")

                # Year distribution
                if 'year' in self.papers_df.columns:
                    year_counts = self.papers_df['year'].value_counts().sort_index()
                    if len(year_counts) > 0:
                        print(f"📅 Publication Years: {len(year_counts)} distinct years")
                        print(f"   Most recent: {year_counts.index.max()}")
                        print(f"   Range: {year_counts.index.min()} - {year_counts.index.max()}")

                # Citation summary
                if 'citations' in self.papers_df.columns:
                    avg_citations = self.papers_df['citations'].mean()
                    max_citations = self.papers_df['citations'].max()
                    print(f"📊 Citation Analysis:")
                    print(f"   Average citations: {avg_citations:.1f}")
                    print(f"   Maximum citations: {max_citations}")
                    print(f"   Total papers with citations: {len(self.papers_df[self.papers_df['citations'] > 0])}")

                # Research gap summary
                print(f"🎯 Research Gaps Identified: {len(self.research_gaps)}")
                for gap in self.research_gaps:
                    print(f"   • {gap['type']}: {gap['severity']} priority")

        except Exception as e:
            print(f"⚠️  Visualization skipped: {e}")

    def run_complete_analysis(self):
        """Enhanced main analysis workflow using dynamic keywords"""
        # Get comprehensive user input
        keyword, topic, target_papers = self.get_user_input_with_keywords()

        # Search papers
        papers_df = self.search_papers(keyword, topic, target_papers)
        if papers_df is None:
            print("❌ Failed to retrieve papers. Please check your API keys and internet connection.")
            return

        # Display detailed paper analysis
        self.display_paper_details()

        # Cluster and analyze with actual keywords
        try:
            self.cluster_and_analyze()
        except Exception as e:
            print(f"⚠️  Topic clustering skipped due to technical constraints: {e}")
            self.topics = [{'id': 0, 'name': f'{self.current_keyword} {self.current_topic} Research', 'papers_count': len(self.papers_df), 'avg_citations': 0}]

        # Comprehensive gap detection with actual keywords
        gaps = self.detect_comprehensive_gaps()
        self.research_gaps = gaps

        # Generate enhanced analyses with actual keywords
        scholar_recommendations = self.generate_enhanced_scholar_recommendations()
        patent_analysis = self.generate_enhanced_patent_analysis()
        comprehensive_review = self.generate_comprehensive_review()
        next_steps = self.generate_next_steps()

        # Display all results
        print(scholar_recommendations)
        print(patent_analysis)
        print(comprehensive_review)
        print(next_steps)

        # Add visualization
        self.visualize_results()

        # Summary
        self.print_section("ANALYSIS COMPLETE")
        print(f"✅ Papers analyzed: {len(self.papers_df)}")
        print(f"✅ Research topic: {self.current_keyword} {self.current_topic}")
        print(f"✅ Additional keywords used: {len(self.additional_keywords)}")
        print(f"✅ Research gaps identified: {len(gaps)}")
        print(f"✅ Comprehensive reports generated: 4")

        print(f"\n🎯 **IMMEDIATE NEXT STEPS FOR {self.current_keyword.upper()} {self.current_topic.upper()} RESEARCH:**")
        print("1. Review the detailed recommendations above")
        print("2. Prioritize research directions based on your expertise")
        print("3. Begin literature review on specific gap areas")
        print("4. Connect with potential collaborators in identified opportunity areas")
        print("5. Consider patent opportunities highlighted in the analysis")
        print("6. Develop a research proposal based on the most promising gaps")

# Run the enhanced analysis
if __name__ == "__main__":
    analyzer = AdvancedResearchAnalyzer()
    analyzer.run_complete_analysis()

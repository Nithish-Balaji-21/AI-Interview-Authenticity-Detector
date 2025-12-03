"""
Comprehensive Fraud Detection System
Multi-modal fraud detection beyond just camera analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import re
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


@dataclass
class FraudIndicator:
    """Represents a detected fraud indicator"""
    name: str
    severity: float  # 0-1, where 1 is most severe
    confidence: float  # 0-1, how confident we are in this indicator
    description: str
    evidence: List[str] = field(default_factory=list)
    category: str = "behavioral"  # behavioral, textual, timing, technical
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'severity': self.severity,
            'confidence': self.confidence,
            'description': self.description,
            'evidence': self.evidence,
            'category': self.category
        }


class ComprehensiveFraudDetector:
    """
    Multi-modal fraud detection system
    Combines behavioral, textual, timing, and technical analysis
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize fraud detector
        
        Args:
            model_path: Path to trained behavioral model
        """
        self.behavioral_model = None
        self.scaler = None
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        
        self.fraud_indicators: List[FraudIndicator] = []
        self.analysis_history: List[Dict] = []
        
        logger.info("ComprehensiveFraudDetector initialized")
    
    def _load_model(self, model_path: Path):
        """Load trained behavioral model"""
        try:
            saved_data = joblib.load(model_path)
            self.behavioral_model = saved_data['ensemble']
            self.scaler = saved_data['scaler']
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def reset_indicators(self):
        """Clear all fraud indicators"""
        self.fraud_indicators = []
    
    def analyze_behavioral_patterns(self, features: Dict[str, float]) -> FraudIndicator:
        """
        Analyze behavioral cues from video
        
        Args:
            features: Dictionary of extracted behavioral features
            
        Returns:
            FraudIndicator with behavioral analysis results
        """
        evidence = []
        severity = 0.0
        
        # Analyze individual features for suspicious patterns
        
        # Eye movement analysis
        eye_movement = features.get('eye_movement_freq', 0)
        if eye_movement > 6.0:
            evidence.append(f"High eye movement frequency: {eye_movement:.1f}/s (reading from screen)")
            severity += 0.2
        elif eye_movement < 1.5:
            evidence.append(f"Very low eye movement: {eye_movement:.1f}/s (unnatural stillness)")
            severity += 0.15
        
        # Gaze fixation
        fixation = features.get('eye_fixation_duration', 0)
        if fixation < 200:
            evidence.append(f"Short eye fixation: {fixation:.0f}ms (scanning/reading)")
            severity += 0.15
        
        # Head stability (too stable is suspicious)
        head_stability = features.get('head_stability', 0)
        if head_stability > 0.85:
            evidence.append(f"Unnaturally stable head position: {head_stability:.2f}")
            severity += 0.2
        
        # Response delay
        response_delay = features.get('response_delay', 0)
        if response_delay > 3.5:
            evidence.append(f"Long response delay: {response_delay:.1f}s (possible AI consultation)")
            severity += 0.25
        
        # Emotion analysis
        emotion_stability = features.get('emotion_stability', 0)
        if emotion_stability > 0.85:
            evidence.append(f"Overly stable emotions: {emotion_stability:.2f} (lack of natural variation)")
            severity += 0.15
        
        # Blink rate
        blink_rate = features.get('blink_rate', 0)
        if blink_rate < 12:
            evidence.append(f"Low blink rate: {blink_rate:.0f}/min (high concentration/reading)")
            severity += 0.1
        
        # Cognitive load
        cognitive_load = features.get('cognitive_load_score', 0)
        if cognitive_load < 0.4:
            evidence.append(f"Low cognitive load: {cognitive_load:.2f} (not thinking, just reading)")
            severity += 0.15
        
        # Use ML model if available
        ml_confidence = 0.5
        if self.behavioral_model and self.scaler:
            try:
                import config
                feature_vector = np.array([features.get(name, 0) for name in config.FEATURE_NAMES]).reshape(1, -1)
                feature_vector = self.scaler.transform(feature_vector)
                
                prediction = self.behavioral_model.predict(feature_vector)[0]
                probabilities = self.behavioral_model.predict_proba(feature_vector)[0]
                ml_confidence = probabilities[prediction]
                
                if prediction == 1:  # AI-assisted
                    severity = max(severity, probabilities[1])
                    evidence.append(f"ML Model prediction: AI-Assisted (confidence: {probabilities[1]:.2%})")
            except Exception as e:
                logger.warning(f"Error in ML prediction: {e}")
        
        # Cap severity at 1.0
        severity = min(severity, 1.0)
        
        indicator = FraudIndicator(
            name="Behavioral Analysis",
            severity=severity,
            confidence=ml_confidence,
            description="Analysis of eye movements, head pose, emotions, and response patterns",
            evidence=evidence if evidence else ["No significant behavioral anomalies detected"],
            category="behavioral"
        )
        
        self.fraud_indicators.append(indicator)
        return indicator
    
    def analyze_response_patterns(self, responses: List[str], 
                                  questions: Optional[List[str]] = None) -> FraudIndicator:
        """
        Analyze text responses for AI generation patterns
        
        Args:
            responses: List of candidate responses
            questions: Optional list of questions asked
            
        Returns:
            FraudIndicator with text analysis results
        """
        evidence = []
        ai_score = 0.0
        total_checks = 0
        
        for i, response in enumerate(responses):
            if not response or len(response.strip()) < 10:
                continue
            
            # Check for overly structured responses
            if self._check_overly_structured(response):
                evidence.append(f"Response {i+1}: Overly structured format (numbered lists, bullet points)")
                ai_score += 0.2
            total_checks += 1
            
            # Check for AI-typical language
            ai_phrases = self._check_ai_language(response)
            if ai_phrases:
                evidence.append(f"Response {i+1}: AI-typical phrases detected: {', '.join(ai_phrases[:3])}")
                ai_score += 0.25
            
            # Check for generic/templated answers
            if self._check_generic_answer(response):
                evidence.append(f"Response {i+1}: Generic or templated answer detected")
                ai_score += 0.15
            
            # Check for unnatural perfection
            if self._check_unnatural_perfection(response):
                evidence.append(f"Response {i+1}: Unnaturally perfect grammar and structure")
                ai_score += 0.15
            
            # Check response length consistency (AI tends to be consistent)
            if len(responses) > 2:
                lengths = [len(r.split()) for r in responses]
                if np.std(lengths) < 10:  # Very consistent lengths
                    evidence.append("Suspiciously consistent response lengths")
                    ai_score += 0.1
        
        # Calculate final severity
        severity = min(ai_score / max(total_checks, 1), 1.0) if total_checks > 0 else 0.0
        
        indicator = FraudIndicator(
            name="Response Pattern Analysis",
            severity=severity,
            confidence=0.75,  # Text analysis is fairly reliable
            description="Analysis of text responses for AI generation patterns",
            evidence=evidence if evidence else ["No AI-typical patterns detected in responses"],
            category="textual"
        )
        
        self.fraud_indicators.append(indicator)
        return indicator
    
    def analyze_timing_patterns(self, timestamps: List[float], 
                               question_complexities: Optional[List[str]] = None) -> FraudIndicator:
        """
        Analyze response timing for anomalies
        
        Args:
            timestamps: List of response timestamps
            question_complexities: Optional complexity ratings ('easy', 'medium', 'hard')
            
        Returns:
            FraudIndicator with timing analysis results
        """
        evidence = []
        severity = 0.0
        
        if len(timestamps) < 2:
            return FraudIndicator(
                name="Timing Analysis",
                severity=0.0,
                confidence=0.3,
                description="Insufficient data for timing analysis",
                evidence=["Need at least 2 responses for timing analysis"],
                category="timing"
            )
        
        # Calculate response intervals
        intervals = np.diff(timestamps)
        
        # Check for suspiciously consistent timing
        if len(intervals) > 2:
            std_dev = np.std(intervals)
            if std_dev < 1.0:  # Very consistent
                evidence.append(f"Unnaturally consistent response times (std: {std_dev:.2f}s)")
                severity += 0.3
        
        # Check for very fast responses
        fast_responses = sum(1 for interval in intervals if interval < 5.0)
        if fast_responses > len(intervals) * 0.7:  # More than 70% fast
            evidence.append(f"{fast_responses}/{len(intervals)} suspiciously fast responses (< 5s)")
            severity += 0.25
        
        # Check for very slow responses (might be typing out AI answer)
        slow_responses = sum(1 for interval in intervals if interval > 15.0)
        if slow_responses > len(intervals) * 0.3:
            evidence.append(f"{slow_responses}/{len(intervals)} unusually slow responses (> 15s)")
            severity += 0.15
        
        # Pattern detection: AI users tend to have bimodal distribution
        # (fast copy-paste vs slow generation)
        if len(intervals) > 4:
            q1, q3 = np.percentile(intervals, [25, 75])
            iqr = q3 - q1
            if iqr > 10:  # Large variance
                evidence.append(f"High timing variance (IQR: {iqr:.1f}s) - possible AI consultation")
                severity += 0.2
        
        severity = min(severity, 1.0)
        
        indicator = FraudIndicator(
            name="Timing Analysis",
            severity=severity,
            confidence=0.7,
            description="Analysis of response timing patterns",
            evidence=evidence if evidence else ["Normal timing patterns detected"],
            category="timing"
        )
        
        self.fraud_indicators.append(indicator)
        return indicator
    
    def analyze_screen_activity(self, screen_data: Dict) -> Optional[FraudIndicator]:
        """
        Analyze screen/browser activity for fraud indicators
        
        Args:
            screen_data: Dictionary with screen activity data
                - tab_switches: Number of tab switches
                - copy_paste_events: Number of copy-paste actions
                - external_tools_detected: List of detected tools
                - window_focus_changes: Number of window focus changes
                
        Returns:
            FraudIndicator if suspicious activity detected, None otherwise
        """
        evidence = []
        severity = 0.0
        
        # Tab switching
        tab_switches = screen_data.get('tab_switches', 0)
        if tab_switches > 3:
            evidence.append(f"Multiple tab switches detected: {tab_switches}")
            severity += min(tab_switches * 0.15, 0.5)
        
        # Copy-paste activity
        copy_paste = screen_data.get('copy_paste_events', 0)
        if copy_paste > 0:
            evidence.append(f"Copy-paste activity detected: {copy_paste} times")
            severity += min(copy_paste * 0.2, 0.6)
        
        # External tools
        external_tools = screen_data.get('external_tools_detected', [])
        if external_tools:
            evidence.append(f"External AI tools detected: {', '.join(external_tools)}")
            severity = 0.95  # Very high severity if AI tools detected
        
        # Window focus changes
        focus_changes = screen_data.get('window_focus_changes', 0)
        if focus_changes > 5:
            evidence.append(f"Frequent window switches: {focus_changes}")
            severity += min(focus_changes * 0.1, 0.4)
        
        # Browser activity
        if 'chatgpt_detected' in screen_data and screen_data['chatgpt_detected']:
            evidence.append("ChatGPT usage detected during interview")
            severity = 0.98
        
        if not evidence:
            return None
        
        severity = min(severity, 1.0)
        
        indicator = FraudIndicator(
            name="Screen Activity Analysis",
            severity=severity,
            confidence=1.0,  # Screen activity is very reliable
            description="Analysis of screen and browser activity",
            evidence=evidence,
            category="technical"
        )
        
        self.fraud_indicators.append(indicator)
        return indicator
    
    def get_comprehensive_assessment(self) -> Dict:
        """
        Get overall fraud assessment from all indicators
        
        Returns:
            Dictionary with comprehensive assessment
        """
        if not self.fraud_indicators:
            return {
                'overall_risk': 'LOW',
                'risk_score': 0.0,
                'confidence': 0.0,
                'indicators': [],
                'recommendation': 'No analysis performed yet.',
                'summary': 'No fraud indicators detected.'
            }
        
        # Calculate weighted risk score
        total_weighted_severity = sum(
            ind.severity * ind.confidence 
            for ind in self.fraud_indicators
        )
        total_weight = sum(ind.confidence for ind in self.fraud_indicators)
        
        risk_score = total_weighted_severity / total_weight if total_weight > 0 else 0.0
        avg_confidence = np.mean([ind.confidence for ind in self.fraud_indicators])
        
        # Determine risk level
        risk_level = self._calculate_risk_level(risk_score)
        
        # Generate summary
        summary = self._generate_summary()
        
        # Store in history
        assessment = {
            'overall_risk': risk_level,
            'risk_score': risk_score,
            'confidence': avg_confidence,
            'indicators': [ind.to_dict() for ind in self.fraud_indicators],
            'recommendation': self._get_recommendation(risk_level, risk_score),
            'summary': summary,
            'num_indicators': len(self.fraud_indicators),
            'categories_analyzed': list(set(ind.category for ind in self.fraud_indicators))
        }
        
        self.analysis_history.append(assessment)
        
        return assessment
    
    def _calculate_risk_level(self, risk_score: float) -> str:
        """Calculate risk level from risk score"""
        if risk_score >= 0.75:
            return 'CRITICAL'
        elif risk_score >= 0.55:
            return 'HIGH'
        elif risk_score >= 0.35:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_summary(self) -> str:
        """Generate human-readable summary of findings"""
        if not self.fraud_indicators:
            return "No fraud indicators detected."
        
        high_severity = [ind for ind in self.fraud_indicators if ind.severity > 0.6]
        
        if high_severity:
            categories = set(ind.category for ind in high_severity)
            return f"Detected {len(high_severity)} high-severity indicators across {len(categories)} categories. " \
                   f"Primary concerns: {', '.join(ind.name for ind in high_severity[:3])}."
        
        medium_severity = [ind for ind in self.fraud_indicators if 0.3 < ind.severity <= 0.6]
        if medium_severity:
            return f"Detected {len(medium_severity)} medium-severity indicators. Further review recommended."
        
        return "Minor indicators detected. Behavior appears mostly genuine."
    
    def _get_recommendation(self, risk_level: str, risk_score: float) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            'LOW': '✅ No significant fraud indicators detected. Candidate appears genuine. Proceed with standard evaluation.',
            'MEDIUM': '⚠️ Some suspicious indicators detected. Recommend additional verification questions or manual review of specific responses.',
            'HIGH': '🚨 Multiple fraud indicators detected. Strongly recommend thorough manual review, additional verification, and possibly re-interview under controlled conditions.',
            'CRITICAL': '🛑 Critical fraud indicators detected. Strong evidence of AI assistance or cheating. Recommend immediate disqualification or formal investigation.'
        }
        
        base_rec = recommendations.get(risk_level, 'Unknown risk level')
        
        # Add specific actions based on indicators
        actions = []
        for ind in self.fraud_indicators:
            if ind.severity > 0.7:
                if ind.category == 'technical':
                    actions.append("Implement stricter browser monitoring")
                elif ind.category == 'behavioral':
                    actions.append("Request live video re-interview")
                elif ind.category == 'textual':
                    actions.append("Ask follow-up questions about responses")
        
        if actions:
            base_rec += f"\n\nSuggested actions: {', '.join(set(actions))}."
        
        return base_rec
    
    # Helper methods for text analysis
    
    def _check_overly_structured(self, text: str) -> bool:
        """Check for overly structured responses (AI tendency)"""
        structure_indicators = [
            r'\d+\.',  # Numbered lists
            r'•',      # Bullet points
            r'\n-\s',  # Dash lists
            r'First,',
            r'Second,',
            r'Third,',
            r'Finally,',
            r'In conclusion',
            r'To summarize'
        ]
        
        matches = sum(1 for indicator in structure_indicators 
                     if re.search(indicator, text, re.IGNORECASE))
        
        return matches >= 3
    
    def _check_ai_language(self, text: str) -> List[str]:
        """Check for AI-typical language patterns"""
        ai_phrases = [
            'as an ai', 'i apologize', "it's important to note",
            'however, it\'s worth', 'in conclusion', 'to summarize',
            'it is crucial to', 'it is essential to',
            'furthermore', 'moreover', 'in addition to',
            'comprehensive approach', 'holistic perspective',
            'it\'s worth noting', 'that being said',
            'leverage', 'utilize' , 'paradigm'
        ]
        
        found_phrases = []
        text_lower = text.lower()
        
        for phrase in ai_phrases:
            if phrase in text_lower:
                found_phrases.append(phrase)
        
        return found_phrases
    
    def _check_generic_answer(self, text: str) -> bool:
        """Check for generic/templated answers"""
        # Check vocabulary diversity
        words = text.split()
        if len(words) < 20:
            return False
        
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        # Low diversity suggests templated response
        return diversity_ratio < 0.55
    
    def _check_unnatural_perfection(self, text: str) -> bool:
        """Check for unnaturally perfect grammar (AI tendency)"""
        # Check for common human speech patterns
        human_patterns = ['um', 'uh', 'like', 'you know', 'i mean', 
                         'kind of', 'sort of', 'basically']
        
        # AI responses rarely have filler words
        has_fillers = any(pattern in text.lower() for pattern in human_patterns)
        
        # Check for contractions (humans use them more)
        contractions = ["i'm", "i've", "it's", "that's", "don't", "can't", "won't"]
        has_contractions = any(cont in text.lower() for cont in contractions)
        
        # AI tends to have perfect punctuation and no fillers
        return not has_fillers and not has_contractions and len(text.split()) > 30


# Convenience function for quick analysis
def quick_fraud_check(features: Dict[str, float], 
                     model_path: Optional[Path] = None) -> Dict:
    """
    Quick fraud check on behavioral features
    
    Args:
        features: Dictionary of behavioral features
        model_path: Optional path to trained model
        
    Returns:
        Assessment dictionary
    """
    detector = ComprehensiveFraudDetector(model_path)
    detector.analyze_behavioral_patterns(features)
    return detector.get_comprehensive_assessment()

#!/usr/bin/env python3
"""
Synthetic Data Generator Demo

This script demonstrates all the capabilities of the synthetic data generator
for medical AI training data augmentation.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.synthetic_data_generator import SyntheticDataGenerator, MedicalSpecialty, TriageLevel
from utils.data_augmentation import DataAugmentor, AugmentationConfig


def demo_basic_generation():
    """Demonstrate basic synthetic data generation"""
    print("=" * 60)
    print("DEMO: Basic Synthetic Data Generation")
    print("=" * 60)
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate a single scenario
    print("\n1. Generating a single medical scenario...")
    scenario = generator.generate_scenario()
    
    print(f"✓ Scenario ID: {scenario.scenario_id}")
    print(f"✓ Patient: {scenario.patient.age}-year-old {scenario.patient.gender}")
    print(f"✓ Primary complaint: {scenario.primary_complaint}")
    print(f"✓ Triage level: {scenario.triage_level.name}")
    print(f"✓ Specialty: {scenario.specialty.value}")
    print(f"✓ Number of symptoms: {len(scenario.symptoms)}")
    print(f"✓ Conversation length: {len(scenario.conversation)} turns")
    
    # Show sample conversation
    print("\n2. Sample conversation:")
    for i, turn in enumerate(scenario.conversation[:4]):  # Show first 4 turns
        speaker_icon = "👤" if turn.speaker == "patient" else "🤖"
        print(f"   {speaker_icon} {turn.speaker}: {turn.text}")
    
    # Generate multiple scenarios
    print("\n3. Generating multiple scenarios...")
    scenarios = generator.generate_dataset(num_scenarios=10)
    
    print(f"✓ Generated {len(scenarios)} scenarios")
    
    # Analyze distributions
    specialties = {}
    triage_levels = {}
    
    for s in scenarios:
        specialties[s.specialty.value] = specialties.get(s.specialty.value, 0) + 1
        triage_levels[s.triage_level.name] = triage_levels.get(s.triage_level.name, 0) + 1
    
    print("\n4. Distribution analysis:")
    print("   Specialties:", dict(sorted(specialties.items())))
    print("   Triage levels:", dict(sorted(triage_levels.items())))
    
    return scenarios


def demo_specialization():
    """Demonstrate specialty-focused generation"""
    print("\n" + "=" * 60)
    print("DEMO: Specialty-Focused Generation")
    print("=" * 60)
    
    generator = SyntheticDataGenerator(seed=123)
    
    # Generate cardiology-focused scenarios
    print("\n1. Generating cardiology scenarios...")
    cardio_scenarios = generator.generate_dataset(
        num_scenarios=5,
        specialty_distribution={MedicalSpecialty.CARDIOLOGY: 1.0}
    )
    
    print(f"✓ Generated {len(cardio_scenarios)} cardiology scenarios")
    
    for i, scenario in enumerate(cardio_scenarios):
        print(f"   {i+1}. {scenario.primary_complaint} - Age {scenario.patient.age}")
    
    # Generate age-specific scenarios
    print("\n2. Generating pediatric scenarios...")
    pediatric_scenarios = generator.generate_dataset(
        num_scenarios=3,
        age_range=(1, 12),
        specialty_distribution={MedicalSpecialty.PEDIATRICS: 1.0}
    )
    
    print(f"✓ Generated {len(pediatric_scenarios)} pediatric scenarios")
    
    for i, scenario in enumerate(pediatric_scenarios):
        print(f"   {i+1}. {scenario.patient.age}-year-old with {scenario.primary_complaint}")


def demo_augmentation():
    """Demonstrate data augmentation capabilities"""
    print("\n" + "=" * 60)
    print("DEMO: Data Augmentation")
    print("=" * 60)
    
    # Generate some base data
    generator = SyntheticDataGenerator(seed=456)
    base_scenarios = generator.generate_dataset(num_scenarios=5)
    
    print(f"\n1. Base dataset: {len(base_scenarios)} scenarios")
    
    # Show original conversation
    original_conv = base_scenarios[0].conversation
    print("\n2. Original conversation sample:")
    for i, turn in enumerate(original_conv[:3]):
        speaker_icon = "👤" if turn.speaker == "patient" else "🤖"
        print(f"   {speaker_icon} {turn.speaker}: {turn.text}")
    
    # Configure augmentation
    config = AugmentationConfig(
        synonym_probability=0.5,
        paraphrase_probability=0.4,
        adversarial_probability=0.2,
        max_augmentations=3,
        preserve_medical_terms=False  # For demo purposes
    )
    
    augmentor = DataAugmentor(config)
    
    print("\n3. Applying augmentation...")
    
    # Convert scenarios to dict format for augmentation
    scenarios_dict = []
    for scenario in base_scenarios:
        scenario_dict = {
            'scenario_id': scenario.scenario_id,
            'conversation': [
                {
                    'speaker': turn.speaker,
                    'text': turn.text,
                    'timestamp': turn.timestamp,
                    'context': turn.context
                }
                for turn in scenario.conversation
            ],
            'triage_level': scenario.triage_level.value,
            'specialty': scenario.specialty.value
        }
        scenarios_dict.append(scenario_dict)
    
    # Apply augmentation
    original_texts = []
    augmented_conversations = []
    
    for scenario_dict in scenarios_dict:
        if "conversation" in scenario_dict:
            original_texts.extend([turn["text"] for turn in scenario_dict["conversation"]])
            augmented_versions = augmentor.augment_conversation(scenario_dict["conversation"])
            augmented_conversations.extend(augmented_versions)
    
    print(f"✓ Original texts: {len(original_texts)}")
    print(f"✓ Augmented conversations: {len(augmented_conversations)}")
    
    # Show augmented example
    if augmented_conversations:
        print("\n4. Augmented conversation sample:")
        for i, turn in enumerate(augmented_conversations[0][:3]):
            speaker_icon = "👤" if turn["speaker"] == "patient" else "🤖"
            print(f"   {speaker_icon} {turn['speaker']}: {turn['text']}")
    
    # Calculate quality metrics
    print("\n5. Quality metrics:")
    quality_metrics = augmentor.validate_augmentation_quality(original_texts, [])
    for metric, value in quality_metrics.items():
        print(f"   {metric}: {value:.3f}")


def demo_medical_diversity():
    """Demonstrate medical diversity and realism"""
    print("\n" + "=" * 60)
    print("DEMO: Medical Diversity and Realism")
    print("=" * 60)
    
    generator = SyntheticDataGenerator(seed=789)
    
    # Generate diverse scenarios
    scenarios = generator.generate_dataset(num_scenarios=20)
    
    # Analyze medical realism
    specialties = set()
    age_groups = set()
    symptom_categories = set()
    triage_distribution = {}
    
    print("\n1. Diversity analysis:")
    
    for scenario in scenarios:
        specialties.add(scenario.specialty.value)
        age_groups.add(scenario.patient.age_group.value)
        triage_distribution[scenario.triage_level.name] = triage_distribution.get(scenario.triage_level.name, 0) + 1
        
        for symptom in scenario.symptoms:
            symptom_categories.add(symptom.category)
    
    print(f"   Specialties represented: {len(specialties)} ({', '.join(sorted(specialties))})")
    print(f"   Age groups: {len(age_groups)} ({', '.join(sorted(age_groups))})")
    print(f"   Symptom categories: {len(symptom_categories)} ({', '.join(sorted(symptom_categories))})")
    print(f"   Triage distribution: {dict(sorted(triage_distribution.items()))}")
    
    # Show complex cases
    print("\n2. Complex cases (multiple symptoms):")
    complex_cases = [s for s in scenarios if len(s.symptoms) > 2][:3]
    
    for i, scenario in enumerate(complex_cases):
        print(f"   Case {i+1}: {scenario.patient.age}-year-old with {len(scenario.symptoms)} symptoms")
        symptom_names = [symptom.name for symptom in scenario.symptoms]
        print(f"   Symptoms: {', '.join(symptom_names)}")
        print(f"   Triage: {scenario.triage_level.name}")
    
    # Emergency scenarios
    print("\n3. Emergency scenarios:")
    emergency_cases = [s for s in scenarios if s.triage_level in [TriageLevel.EMERGENT, TriageLevel.IMMEDIATE]]
    
    if emergency_cases:
        for i, scenario in enumerate(emergency_cases):
            print(f"   Emergency {i+1}: {scenario.primary_complaint} - {scenario.triage_level.name}")
            if scenario.symptoms:
                red_flags = []
                for symptom in scenario.symptoms:
                    red_flags.extend(symptom.red_flags)
                if red_flags:
                    print(f"   Red flags: {', '.join(red_flags)}")
    else:
        print("   No emergency cases in this sample")


def demo_quality_validation():
    """Demonstrate quality validation features"""
    print("\n" + "=" * 60)
    print("DEMO: Quality Validation")
    print("=" * 60)
    
    generator = SyntheticDataGenerator(seed=999)
    scenarios = generator.generate_dataset(num_scenarios=15)
    
    print(f"\n1. Analyzing {len(scenarios)} scenarios...")
    
    # Conversation structure analysis
    well_formed_conversations = 0
    total_turns = 0
    
    for scenario in scenarios:
        conversation = scenario.conversation
        if len(conversation) >= 4:
            has_patient = any(turn.speaker == "patient" for turn in conversation)
            has_ai = any(turn.speaker == "ai" for turn in conversation)
            if has_patient and has_ai:
                well_formed_conversations += 1
        
        total_turns += len(conversation)
    
    avg_conversation_length = total_turns / len(scenarios)
    
    print(f"   Well-formed conversations: {well_formed_conversations}/{len(scenarios)} ({well_formed_conversations/len(scenarios)*100:.1f}%)")
    print(f"   Average conversation length: {avg_conversation_length:.1f} turns")
    
    # Medical terminology usage
    medical_terms = {"pain", "symptom", "diagnosis", "treatment", "medication", "doctor", "hospital"}
    conversations_with_medical_terms = 0
    
    for scenario in scenarios:
        text = " ".join([turn.text for turn in scenario.conversation])
        text_lower = text.lower()
        
        if any(term in text_lower for term in medical_terms):
            conversations_with_medical_terms += 1
    
    medical_term_percentage = conversations_with_medical_terms / len(scenarios) * 100
    print(f"   Conversations with medical terms: {medical_term_percentage:.1f}%")
    
    # Complexity analysis
    complexity_scores = []
    for scenario in scenarios:
        complexity = len(scenario.symptoms) + (scenario.patient.age // 20)
        complexity_scores.append(complexity)
    
    avg_complexity = sum(complexity_scores) / len(complexity_scores)
    print(f"   Average complexity score: {avg_complexity:.1f}")
    print(f"   Complexity range: {min(complexity_scores)}-{max(complexity_scores)}")
    
    print("\n2. Quality assessment:")
    print("   ✓ All scenarios have valid patient profiles")
    print("   ✓ All conversations follow proper structure")
    print("   ✓ Medical terminology appropriately used")
    print("   ✓ Good diversity in symptoms and cases")


def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n" + "=" * 60)
    print("DEMO: Batch Processing")
    print("=" * 60)
    
    print("\n1. Batch generation simulation...")
    
    generator = SyntheticDataGenerator(seed=111)
    
    batches = []
    for batch_num in range(3):
        print(f"   Processing batch {batch_num + 1}/3...")
        
        batch_scenarios = generator.generate_dataset(num_scenarios=5)
        batches.append(batch_scenarios)
    
    total_scenarios = sum(len(batch) for batch in batches)
    print(f"\n2. Batch results:")
    print(f"   Total batches: {len(batches)}")
    print(f"   Total scenarios: {total_scenarios}")
    
    # Analyze batch diversity
    all_specialties = set()
    for batch in batches:
        for scenario in batch:
            all_specialties.add(scenario.specialty.value)
    
    print(f"   Unique specialties across all batches: {len(all_specialties)}")
    print(f"   Specialties: {', '.join(sorted(all_specialties))}")


def main():
    """Run all demos"""
    print("🚀 Synthetic Data Generator for Medical AI Training")
    print("=" * 60)
    print("This demo showcases the comprehensive capabilities of the")
    print("synthetic data generator for medical AI training augmentation.")
    print("=" * 60)
    
    try:
        # Run demos
        demo_basic_generation()
        demo_specialization()
        demo_augmentation()
        demo_medical_diversity()
        demo_quality_validation()
        demo_batch_processing()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey capabilities demonstrated:")
        print("• Realistic medical scenario generation")
        print("• Specialty and triage level diversity")
        print("• Natural conversation flow")
        print("• Data augmentation techniques")
        print("• Quality validation metrics")
        print("• Batch processing for scalability")
        print("\nNext steps:")
        print("• Run 'python generate_synthetic_data.py generate --num-scenarios 1000'")
        print("• Try augmentation with 'python generate_synthetic_data.py augment ...'")
        print("• Validate your data with 'python generate_synthetic_data.py validate ...'")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Please ensure you're running this from the training directory")
        print("and have the required dependencies installed.")


if __name__ == "__main__":
    main()
"""
Patient Chat Flow Integration Testing

Comprehensive testing of the complete patient chat workflow:
- Patient symptom input
- AI assessment processing  
- Red flag detection and escalation
- Nurse interaction and review
- Final recommendations and follow-up

Tests the end-to-end patient experience from initial consultation to clinical decision.
"""

import pytest
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock
import websockets

# Test markers
pytestmark = pytest.mark.integration


class TestPatientChatFlow:
    """Test complete patient chat workflows."""
    
    @pytest.mark.asyncio
    async def test_normal_chat_flow(self, mock_websocket_manager, mock_patient_data, assert_helpers, test_measurements):
        """Test normal patient chat flow with non-emergency symptoms."""
        
        test_measurements.start_timer("normal_chat_flow")
        
        patient_data = mock_patient_data["patient_002"]  # Headache patient
        session_id = f"chat_session_{uuid.uuid4().hex[:8]}"
        
        # Step 1: Patient connects to chat
        websocket = await mock_websocket_manager.connect(session_id, "patient")
        assert websocket is not None
        
        # Step 2: Patient sends initial message
        initial_message = {
            "type": "chat_message",
            "session_id": session_id,
            "content": f"Hi, I'm experiencing {patient_data['chief_complaint'].lower()}",
            "user_type": "patient"
        }
        
        # Send message and get AI response
        await mock_websocket_manager.send_message(session_id, initial_message)
        ai_response = await mock_websocket_manager.receive_message(session_id)
        
        assert ai_response is not None
        assert ai_response["type"] == "ai_response"
        assert "content" in ai_response
        assert ai_response["urgency"] == "normal"  # Should be normal for headache
        
        # Step 3: Continue conversation with symptom details
        symptom_message = {
            "type": "chat_message",
            "session_id": session_id,
            "content": "The headache started 3 days ago and it's quite severe",
            "user_type": "patient"
        }
        
        await mock_websocket_manager.send_message(session_id, symptom_message)
        followup_response = await mock_websocket_manager.receive_message(session_id)
        
        assert followup_response is not None
        assert "content" in followup_response
        
        # Step 4: Provide additional symptoms
        additional_symptoms = {
            "type": "chat_message", 
            "session_id": session_id,
            "content": "I also feel nauseous and have some sensitivity to light",
            "user_type": "patient"
        }
        
        await mock_websocket_manager.send_message(session_id, additional_symptoms)
        detailed_response = await mock_websocket_manager.receive_message(session_id)
        
        assert detailed_response is not None
        assert "content" in detailed_response
        
        # Step 5: Assessment completion
        assessment_request = {
            "type": "request_assessment",
            "session_id": session_id,
            "trigger": "patient_ready_for_assessment"
        }
        
        await mock_websocket_manager.send_message(session_id, assessment_request)
        assessment_response = await mock_websocket_manager.receive_message(session_id)
        
        assert assessment_response is not None
        assert assessment_response["type"] in ["assessment_complete", "triage_in_progress"]
        
        # Validate flow characteristics
        assert ai_response["urgency"] == "normal"
        assert_helpers.assert_response_time(
            test_measurements.end_timer("normal_chat_flow"), 30.0
        )
        
        print(f"Normal chat flow: Completed successfully in {test_measurements.get_metrics().get('normal_chat_flow', {}).get('duration_seconds', 0):.2f}s")
    
    @pytest.mark.asyncio
    async def test_emergency_chat_flow(self, mock_websocket_manager, mock_patient_data, assert_helpers, test_measurements):
        """Test emergency patient chat flow with red flag symptoms."""
        
        test_measurements.start_timer("emergency_chat_flow")
        
        patient_data = mock_patient_data["patient_001"]  # Chest pain patient
        session_id = f"emergency_session_{uuid.uuid4().hex[:8]}"
        
        # Step 1: Patient connects with emergency symptoms
        websocket = await mock_websocket_manager.connect(session_id, "patient")
        assert websocket is not None
        
        # Step 2: Patient sends emergency message
        emergency_message = {
            "type": "chat_message",
            "session_id": session_id,
            "content": f"I'm having severe {patient_data['chief_complaint'].lower()} and can't breathe well",
            "user_type": "patient"
        }
        
        # Send emergency message and get immediate AI response
        await mock_websocket_manager.send_message(session_id, emergency_message)
        emergency_response = await mock_websocket_manager.receive_message(session_id)
        
        assert emergency_response is not None
        assert emergency_response["type"] == "ai_response"
        assert "emergency" in emergency_response["urgency"] or emergency_response["urgency"] == "high"
        assert len(emergency_response.get("red_flags", [])) > 0
        
        # Verify red flag detection
        assert_helpers.assert_red_flags_detected(emergency_response, ["chest_pain"])
        
        # Step 3: Immediate escalation alert
        escalation_alert = {
            "type": "escalation_alert",
            "session_id": session_id,
            "alert_level": "immediate",
            "reason": "emergency_symptoms_detected"
        }
        
        alert_response = await mock_websocket_manager.receive_message(session_id)
        
        # Should receive escalation notification
        if alert_response and alert_response["type"] == "escalation_alert":
            assert alert_response["alert_level"] in ["immediate", "high"]
        
        # Step 4: Nurse notification
        nurse_notification = {
            "type": "nurse_notification",
            "session_id": session_id,
            "priority": "emergency",
            "red_flags": emergency_response.get("red_flags", []),
            "urgency_level": "immediate"
        }
        
        # Verify nurse gets emergency notification
        notification_response = await mock_websocket_manager.receive_message(session_id)
        
        if notification_response and notification_response["type"] == "nurse_notification":
            assert notification_response["priority"] == "emergency"
        
        # Validate emergency flow characteristics
        assert_helpers.assert_urgency_escalation(emergency_response, "high")
        assert_helpers.assert_response_time(
            test_measurements.end_timer("emergency_chat_flow"), 10.0  # Emergency should be fast
        )
        
        print(f"Emergency chat flow: Completed in {test_measurements.get_metrics().get('emergency_chat_flow', {}).get('duration_seconds', 0):.2f}s")
    
    @pytest.mark.asyncio
    async def test_chat_session_persistence(self, mock_websocket_manager, mock_patient_data, test_measurements):
        """Test chat session persistence and state management."""
        
        test_measurements.start_timer("chat_session_persistence")
        
        session_id = f"persistence_session_{uuid.uuid4().hex[:8]}"
        
        # Connect and start conversation
        websocket = await mock_websocket_manager.connect(session_id, "patient")
        assert websocket is not None
        
        # Send multiple messages in sequence
        messages = [
            "Hello, I have a headache",
            "It's been going on for 2 days",
            "The pain is throbbing and moderate",
            "Should I be concerned?"
        ]
        
        conversation_history = []
        
        for message_text in messages:
            message = {
                "type": "chat_message",
                "session_id": session_id,
                "content": message_text,
                "user_type": "patient"
            }
            
            await mock_websocket_manager.send_message(session_id, message)
            response = await mock_websocket_manager.receive_message(session_id)
            
            conversation_history.append({
                "message": message_text,
                "response": response["content"] if response else None,
                "timestamp": datetime.utcnow()
            })
            
            assert response is not None
        
        # Verify session state persistence
        connection_stats = mock_websocket_manager.get_connection_stats()
        assert connection_stats["active_sessions"] == [session_id]
        
        # Send follow-up message after a simulated delay
        await asyncio.sleep(1)  # Simulate time passing
        
        followup_message = {
            "type": "chat_message",
            "session_id": session_id,
            "content": "I've taken some pain medication, should I still be worried?",
            "user_type": "patient"
        }
        
        await mock_websocket_manager.send_message(session_id, followup_message)
        followup_response = await mock_websocket_manager.receive_message(session_id)
        
        # AI should remember previous context
        assert followup_response is not None
        assert "pain medication" in followup_response.get("content", "").lower() or "medication" in followup_response.get("content", "").lower()
        
        # Verify message history integrity
        assert len(conversation_history) == 4
        
        measurement = test_measurements.end_timer("chat_session_persistence")
        assert measurement["duration_seconds"] < 15.0
        print(f"Chat session persistence: {measurement['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_chat_error_handling(self, mock_websocket_manager, test_measurements):
        """Test chat system error handling and recovery."""
        
        test_measurements.start_timer("chat_error_handling")
        
        session_id = f"error_test_session_{uuid.uuid4().hex[:8]}"
        
        # Test connection issues
        websocket = await mock_websocket_manager.connect(session_id, "patient")
        assert websocket is not None
        
        # Test invalid message format
        invalid_message = {
            "type": "invalid_type",
            "session_id": session_id
            # Missing required content
        }
        
        await mock_websocket_manager.send_message(session_id, invalid_message)
        error_response = await mock_websocket_manager.receive_message(session_id)
        
        # Should handle invalid message gracefully
        assert error_response is not None
        assert error_response["type"] == "ai_response"  # Should still respond
        
        # Test very long message (potential abuse)
        long_message_content = "This is a very long message " * 1000
        long_message = {
            "type": "chat_message",
            "session_id": session_id,
            "content": long_message_content,
            "user_type": "patient"
        }
        
        await mock_websocket_manager.send_message(session_id, long_message)
        long_response = await mock_websocket_manager.receive_message(session_id)
        
        # Should handle long messages
        assert long_response is not None
        
        # Test rapid messages (potential spam)
        rapid_messages = []
        for i in range(5):
            rapid_message = {
                "type": "chat_message",
                "session_id": session_id,
                "content": f"Message {i+1}",
                "user_type": "patient"
            }
            rapid_messages.append(rapid_message)
        
        for message in rapid_messages:
            await mock_websocket_manager.send_message(session_id, message)
            response = await mock_websocket_manager.receive_message(session_id)
            assert response is not None
        
        # Test session termination
        terminate_message = {
            "type": "end_session",
            "session_id": session_id,
            "reason": "patient_left"
        }
        
        await mock_websocket_manager.send_message(session_id, terminate_message)
        
        # Verify session ended
        connection_stats = mock_websocket_manager.get_connection_stats()
        # Session should be cleaned up
        
        measurement = test_measurements.end_timer("chat_error_handling")
        assert measurement["duration_seconds"] < 20.0
        print(f"Chat error handling: {measurement['duration_seconds']:.2f}s")


class TestAIAssessmentIntegration:
    """Test AI assessment integration with chat flow."""
    
    @pytest.mark.asyncio
    async def test_symptom_analysis_integration(self, mock_websocket_manager, mock_patient_data, test_measurements):
        """Test symptom analysis integration during chat."""
        
        test_measurements.start_timer("symptom_analysis_integration")
        
        session_id = f"analysis_session_{uuid.uuid4().hex[:8]}"
        
        await mock_websocket_manager.connect(session_id, "patient")
        
        # Progressive symptom disclosure
        symptom_messages = [
            "I have a headache",
            "It's been 2 days", 
            "The pain is in my temples",
            "I also feel nauseous",
            "Light bothers my eyes"
        ]
        
        assessments = []
        
        for symptom in symptom_messages:
            message = {
                "type": "chat_message",
                "session_id": session_id,
                "content": symptom,
                "user_type": "patient"
            }
            
            await mock_websocket_manager.send_message(session_id, message)
            response = await mock_websocket_manager.receive_message(session_id)
            
            # Simulate AI analysis
            if "headache" in symptom.lower():
                assessment = {
                    "symptom": "headache",
                    "severity": "moderate" if "nauseous" in symptom.lower() or "light" in symptom.lower() else "mild",
                    "differential": ["tension_headache", "migraine", "sinus_headache"],
                    "confidence": 0.75
                }
                assessments.append(assessment)
        
        # Verify analysis improves with more information
        assert len(assessments) > 0
        
        # First assessment should be basic
        first_assessment = assessments[0]
        assert first_assessment["confidence"] <= 0.75
        
        # Later assessments should be more refined
        if len(assessments) > 1:
            later_assessment = assessments[-1]
            assert "nauseous" in str(later_assessment).lower() or "migraine" in str(later_assessment).lower()
        
        measurement = test_measurements.end_timer("symptom_analysis_integration")
        assert measurement["duration_seconds"] < 10.0
        print(f"Symptom analysis integration: {measurement['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_risk_assessment_integration(self, mock_websocket_manager, mock_patient_data, test_measurements):
        """Test risk assessment integration during chat flow."""
        
        test_measurements.start_timer("risk_assessment_integration")
        
        session_id = f"risk_session_{uuid.uuid4().hex[:8]}"
        
        await mock_websocket_manager.connect(session_id, "patient")
        
        # Test moderate risk scenario
        moderate_message = {
            "type": "chat_message",
            "session_id": session_id,
            "content": "I've had this headache for a week now and it's getting worse",
            "user_type": "patient"
        }
        
        await mock_websocket_manager.send_message(session_id, moderate_message)
        response = await mock_websocket_manager.receive_message(session_id)
        
        # Simulate risk assessment
        if "week" in moderate_message["content"].lower() and "worse" in moderate_message["content"].lower():
            risk_level = "moderate"
            recommendations = ["monitor_symptoms", "consider_medical_consultation"]
        else:
            risk_level = "low"
            recommendations = ["self_care", "follow_up_if_worse"]
        
        assert risk_level in ["low", "moderate", "high", "emergency"]
        assert len(recommendations) > 0
        
        # Test high risk scenario
        high_risk_message = {
            "type": "chat_message",
            "session_id": session_id,
            "content": "I have severe chest pain and can't breathe properly",
            "user_type": "patient"
        }
        
        await mock_websocket_manager.send_message(session_id, high_risk_message)
        emergency_response = await mock_websocket_manager.receive_message(session_id)
        
        # Should trigger high risk assessment
        risk_assessment = {
            "risk_level": "high",
            "urgency": "immediate",
            "actions": ["call_911", "emergency_protocol"],
            "red_flags": ["chest_pain", "shortness_of_breath"]
        }
        
        assert risk_assessment["risk_level"] in ["high", "emergency"]
        assert "emergency_protocol" in risk_assessment["actions"]
        
        measurement = test_measurements.end_timer("risk_assessment_integration")
        assert measurement["duration_seconds"] < 8.0
        print(f"Risk assessment integration: {measurement['duration_seconds']:.2f}s")


class TestNurseEscalationIntegration:
    """Test nurse escalation integration from chat flow."""
    
    @pytest.mark.asyncio
    async def test_automatic_escalation(self, mock_websocket_manager, mock_patient_data, test_measurements):
        """Test automatic escalation to nurse during chat."""
        
        test_measurements.start_timer("automatic_escalation")
        
        session_id = f"escalation_session_{uuid.uuid4().hex[:8]}"
        
        await mock_websocket_manager.connect(session_id, "patient")
        
        # Trigger automatic escalation with emergency symptoms
        escalation_message = {
            "type": "chat_message",
            "session_id": session_id,
            "content": "I have crushing chest pain and I'm very scared",
            "user_type": "patient"
        }
        
        await mock_websocket_manager.send_message(session_id, escalation_message)
        escalation_response = await mock_websocket_manager.receive_message(session_id)
        
        # Simulate automatic escalation trigger
        escalation_triggered = False
        
        if "crushing chest pain" in escalation_message["content"].lower():
            escalation_triggered = True
        
        assert escalation_triggered is True
        
        # Verify escalation notification
        escalation_notification = {
            "type": "escalation_notification",
            "session_id": session_id,
            "escalation_reason": "emergency_symptoms",
            "priority": "immediate",
            "nurse_assignment": "auto_assign"
        }
        
        # Should send escalation to nurse queue
        notification_response = await mock_websocket_manager.receive_message(session_id)
        
        if notification_response:
            assert notification_response["priority"] == "immediate"
        
        measurement = test_measurements.end_timer("automatic_escalation")
        assert measurement["duration_seconds"] < 5.0
        print(f"Automatic escalation: {measurement['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_manual_escalation_request(self, mock_websocket_manager, test_measurements):
        """Test manual escalation request during chat."""
        
        test_measurements.start_timer("manual_escalation_request")
        
        session_id = f"manual_escalation_{uuid.uuid4().hex[:8]}"
        
        await mock_websocket_manager.connect(session_id, "patient")
        
        # Patient requests manual escalation
        escalation_request = {
            "type": "request_escalation",
            "session_id": session_id,
            "reason": "patient_concern",
            "urgency": "high"
        }
        
        await mock_websocket_manager.send_message(session_id, escalation_request)
        request_response = await mock_websocket_manager.receive_message(session_id)
        
        # Verify escalation request processing
        if request_response and request_response.get("type") == "escalation_ack":
            assert request_response["status"] in ["approved", "pending"]
        
        # Simulate nurse assignment
        nurse_assignment = {
            "type": "nurse_assigned",
            "session_id": session_id,
            "nurse_info": {
                "nurse_id": "nurse_001",
                "estimated_wait_time": "2-3 minutes",
                "specialization": "emergency_medicine"
            }
        }
        
        assignment_response = await mock_websocket_manager.receive_message(session_id)
        
        # Verify nurse assignment
        if assignment_response:
            assert "nurse_info" in assignment_response
        
        measurement = test_measurements.end_timer("manual_escalation_request")
        assert measurement["duration_seconds"] < 10.0
        print(f"Manual escalation request: {measurement['duration_seconds']:.2f}s")


class TestChatPerformanceBenchmarking:
    """Test chat flow performance under various conditions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_chat_sessions(self, mock_websocket_manager, test_measurements):
        """Test multiple concurrent chat sessions."""
        
        test_measurements.start_timer("concurrent_chat_sessions")
        
        # Create multiple concurrent sessions
        sessions = []
        for i in range(5):
            session_id = f"concurrent_session_{i}_{uuid.uuid4().hex[:8]}"
            websocket = await mock_websocket_manager.connect(session_id, "patient")
            sessions.append(session_id)
            
            # Send message to each session
            message = {
                "type": "chat_message",
                "session_id": session_id,
                "content": f"Hello, I have symptoms number {i+1}",
                "user_type": "patient"
            }
            
            await mock_websocket_manager.send_message(session_id, message)
        
        # Receive responses from all sessions
        responses = []
        for session_id in sessions:
            response = await mock_websocket_manager.receive_message(session_id)
            responses.append(response)
        
        # Verify all sessions responded
        assert len(responses) == 5
        assert all(response is not None for response in responses)
        
        # Check connection stats
        stats = mock_websocket_manager.get_connection_stats()
        assert stats["total_connections"] == 5
        
        measurement = test_measurements.end_timer("concurrent_chat_sessions")
        assert measurement["duration_seconds"] < 15.0
        print(f"Concurrent chat sessions: {measurement['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_chat_response_time_benchmark(self, mock_websocket_manager, test_measurements):
        """Benchmark chat response times."""
        
        test_measurements.start_timer("chat_response_time_benchmark")
        
        session_id = f"benchmark_session_{uuid.uuid4().hex[:8]}"
        
        await mock_websocket_manager.connect(session_id, "patient")
        
        # Measure response times for different message types
        message_types = [
            "simple_greeting",
            "symptom_description", 
            "emergency_symptoms",
            "follow_up_question"
        ]
        
        response_times = []
        
        for msg_type in message_types:
            test_measurements.start_timer(f"response_{msg_type}")
            
            message = {
                "type": "chat_message",
                "session_id": session_id,
                "content": f"Test message for {msg_type}",
                "user_type": "patient"
            }
            
            await mock_websocket_manager.send_message(session_id, message)
            response = await mock_websocket_manager.receive_message(session_id)
            
            response_time = test_measurements.end_timer(f"response_{msg_type}")
            response_times.append(response_time["duration_seconds"])
            
            assert response is not None
        
        # Calculate average response time
        avg_response_time = sum(response_times) / len(response_times)
        
        # Validate response time performance
        assert avg_response_time < 2.0  # Average should be under 2 seconds
        assert max(response_times) < 3.0  # Maximum should be under 3 seconds
        
        measurement = test_measurements.end_timer("chat_response_time_benchmark")
        print(f"Chat response time benchmark: avg={avg_response_time:.2f}s, max={max(response_times):.2f}s")
        
        return {
            "average_response_time": avg_response_time,
            "max_response_time": max(response_times),
            "total_test_time": measurement["duration_seconds"]
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
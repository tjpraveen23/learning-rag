# chat_interface.py
import logging
from typing import Dict
from rag_framework import MultiLOBRAGFramework
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatInterface:
    """Chat interface for the Multi-LOB RAG Framework"""
    
    def __init__(self):
        self.rag_framework = MultiLOBRAGFramework()
        self.current_lob = Config.ACTIVE_LOB
        self.conversation_history = []
    
    def start_chat(self):
        """Start interactive chat session"""
        print("🤖 Multi-LOB RAG Assistant")
        print("=" * 50)
        print(f"Available LOBs: {', '.join(self.rag_framework.get_available_lobs())}")
        print(f"Current LOB: {self.current_lob}")
        print("\nCommands:")
        print("  /switch <lob_name> - Switch to different LOB")
        print("  /init <lob_name>   - Initialize specific LOB")
        print("  /init all         - Initialize all LOBs") 
        print("  /status           - Show initialization status")
        print("  /quit             - Exit chat")
        print("=" * 50)
        
        # Initialize current LOB
        print(f"\n🔄 Initializing {self.current_lob}...")
        if self.rag_framework.initialize_lob(self.current_lob):
            print(f"✅ {self.current_lob} initialized successfully")
        else:
            print(f"❌ Failed to initialize {self.current_lob}")
        
        while True:
            try:
                user_input = input(f"\n[{self.current_lob}] Ask a question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == '/quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    self._handle_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat interface: {str(e)}")
                print(f"❌ An error occurred: {str(e)}")
    
    def _handle_command(self, command: str):
        """Handle chat commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/switch':
            if len(parts) > 1:
                new_lob = parts[1]
                if new_lob in self.rag_framework.get_available_lobs():
                    self.current_lob = new_lob
                    print(f"🔄 Switched to LOB: {new_lob}")
                    
                    # Initialize if not already done
                    if new_lob not in self.rag_framework.get_initialized_lobs():
                        print(f"🔄 Initializing {new_lob}...")
                        if self.rag_framework.initialize_lob(new_lob):
                            print(f"✅ {new_lob} initialized successfully")
                        else:
                            print(f"❌ Failed to initialize {new_lob}")
                else:
                    print(f"❌ Invalid LOB. Available: {', '.join(self.rag_framework.get_available_lobs())}")
            else:
                print("❌ Please specify LOB name: /switch <lob_name>")
        
        elif cmd == '/init':
            if len(parts) > 1:
                target = parts[1]
                if target.lower() == 'all':
                    print("🔄 Initializing all LOBs...")
                    results = self.rag_framework.initialize_all_lobs()
                    for lob, success in results.items():
                        status = "✅" if success else "❌"
                        print(f"  {status} {lob}")
                else:
                    print(f"🔄 Initializing {target}...")
                    if self.rag_framework.initialize_lob(target, force_rebuild=True):
                        print(f"✅ {target} initialized successfully")
                    else:
                        print(f"❌ Failed to initialize {target}")
            else:
                print("❌ Please specify LOB name or 'all': /init <lob_name|all>")
        
        elif cmd == '/status':
            available = self.rag_framework.get_available_lobs()
            initialized = self.rag_framework.get_initialized_lobs()
            print(f"\n📊 Status:")
            print(f"  Current LOB: {self.current_lob}")
            print(f"  Available LOBs: {', '.join(available)}")
            print(f"  Initialized LOBs: {', '.join(initialized)}")
            for lob in available:
                status = "✅" if lob in initialized else "❌"
                print(f"    {status} {lob}")
        
        else:
            print("❌ Unknown command. Available commands: /switch, /init, /status, /quit")
    
    def _handle_query(self, question: str):
        """Handle user query"""
        print("🤔 Thinking...")
        
        result = self.rag_framework.query(question, self.current_lob)
        
        print(f"\n📋 Query: {question}")
        print(f"🏢 LOB: {result['lob']}")
        
        if result['success']:
            print(f"✅ Answer:\n{result['answer']}")
            print(f"\n📚 Sources ({result['num_context_docs']} documents):")
            for i, source in enumerate(result['context_sources'], 1):
                print(f"  {i}. {source}")
        else:
            print(f"❌ Error: {result['error']}")
        
        # Add to conversation history
        self.conversation_history.append({
            'question': question,
            'result': result,
            'lob': self.current_lob
        })

from rag_framework import MultiLOBRAGFramework
from chat_interface import ChatInterface

# main.py
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-LOB RAG Framework')
    parser.add_argument('--mode', choices=['chat', 'init'], default='chat',
                      help='Mode: chat for interactive session, init for initialization only')
    parser.add_argument('--lob', type=str, help='Specific LOB to initialize')
    parser.add_argument('--force', action='store_true', help='Force rebuild vector stores')
    
    args = parser.parse_args()
    
    if args.mode == 'init':
        framework = MultiLOBRAGFramework()
        if args.lob:
            print(f"Initializing LOB: {args.lob}")
            success = framework.initialize_lob(args.lob, args.force)
            if success:
                print(f"✅ {args.lob} initialized successfully")
            else:
                print(f"❌ Failed to initialize {args.lob}")
        else:
            print("Initializing all LOBs...")
            results = framework.initialize_all_lobs(args.force)
            for lob, success in results.items():
                status = "✅" if success else "❌"
                print(f"  {status} {lob}")
    else:
        chat = ChatInterface()
        chat.start_chat()

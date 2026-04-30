import argparse
import logging
import os

# Assuming the src directory is in your Python path
from src.data.loader import load_market_data
from src.data.preprocessor import normalize_for_rl
from src.rl.environment import AlphaDiscoveryEnv
from src.rl.agent import AlphaAgent
from src.evaluation.backtester import Backtester
from src.evaluation.factor_analysis import FactorAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("results/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlphaFactoryMain")

def run_data_pipeline(config):
    """Fetches and preprocesses raw market data."""
    logger.info("Starting data pipeline...")
    # Fetch data (e.g., from SQL or MongoDB)
    raw_data = load_market_data(config['tickers'], config['start_date'], config['end_date'])
    
    # Clean and normalize states for the RL agent
    processed_data = normalize_for_rl(raw_data)
    
    # Save processed data to local cache for fast iteration
    processed_data_path = os.path.join("data", "processed", "market_states.pkl")
    processed_data.to_pickle(processed_data_path)
    logger.info(f"Data pipeline complete. Saved to {processed_data_path}")
    return processed_data

# snippet from main.py
def train_agent(data, config):
    """Initializes and trains the reinforcement learning agent."""
    logger.info("Initializing RL Environment...")
    env = AlphaDiscoveryEnv(data=data, max_formula_length=config['max_length'])
    
    logger.info(f"Initializing {config['algorithm']} Agent...")
    agent = AlphaAgent(env=env, algorithm=config['algorithm'])
    
    logger.info(f"Starting training for {config['timesteps']} timesteps...")
    
    # The progress bar is now handled internally by agent.train()
    agent.train(total_timesteps=config['timesteps'])
    
    model_path = os.path.join("models", f"alpha_agent_{config['algorithm']}.zip")
    agent.save(model_path)
    logger.info(f"Training complete. Model saved to {model_path}")
    
    return agent

def evaluate_formulas(agent, env, config):
    """Extracts top formulas from the trained agent and evaluates them."""
    logger.info("Extracting top generated formulas...")
    top_formulas = agent.generate_top_n_formulas(env, n=config['top_n'])
    
    logger.info("Running out-of-sample backtests...")
    backtester = Backtester(out_of_sample_data=config['oos_data'])
    backtest_results = backtester.run(top_formulas)
    
    logger.info("Conducting factor analysis (Value, Momentum, Size, etc.)...")
    analyzer = FactorAnalyzer()
    final_report = analyzer.regress_against_known_factors(backtest_results)
    
    report_path = os.path.join("results", "discovery_report.csv")
    final_report.to_csv(report_path)
    logger.info(f"Evaluation complete. Final report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Automating Alpha Discovery with RL")
    parser.add_argument('--step', choices=['all', 'data', 'train', 'evaluate'], default='all',
                        help="Which step of the pipeline to run.")
    parser.add_argument('--algo', choices=['PPO', 'DQN', 'SAC'], default='PPO',
                        help="RL algorithm to use for the agent.")
    parser.add_argument('--timesteps', type=int, default=100000,
                        help="Number of training timesteps.")
    
    args = parser.parse_args()
    
    # Mock configuration dictionary (typically loaded from a YAML/JSON file)
    config = {
        'tickers': ['AAPL', 'MSFT', 'SPY'],
        'start_date': '2015-01-01',
        'end_date': '2023-01-01',
        'algorithm': args.algo,
        'timesteps': args.timesteps,
        'max_length': 10,
        'top_n': 50,
        'oos_data': 'data/processed/oos_market_states.pkl'
    }

    try:
        if args.step in ['all', 'data']:
            data = run_data_pipeline(config)
        else:
            # If skipping data step, assume processed data exists
            import pandas as pd
            data = pd.read_pickle("data/processed/market_states.pkl")

        if args.step in ['all', 'train']:
            agent = train_agent(data, config)
            
        if args.step in ['all', 'evaluate']:
            # In a real run, you'd load the environment and agent from disk if skipping training
            if args.step == 'evaluate':
                 env = AlphaDiscoveryEnv(data=data, max_formula_length=config['max_length'])
                 agent = AlphaAgent.load(f"models/alpha_agent_{config['algorithm']}.zip", env=env)
            evaluate_formulas(agent, env, config)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Ensure necessary directories exist before running
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    main()
import os
import re
import chess
import chess.engine
import chess.pgn

# Path to Stockfish engine (update this to the correct path on your system)
STOCKFISH_PATH = "./stockfish/stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Path to PGN file (update this to the correct PGN file location)
PGN_PATH = "data/raw_data/lichess_db_standard_rated_2013-01.pgn"

# Output directory for prepared data
OUTPUT_DIRECTORY = "data/prepared_data"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def _get_stockfish_evaluation(board):
    """
    Given a chess.Board object, returns the Stockfish evaluation score in centipawns.
    A return value of None indicates an invalid position (e.g., checkmate).
    """
    info = engine.analyse(board, chess.engine.Limit(time=0.1))  # Time limit for Stockfish analysis
    score = info["score"].relative.score(mate_score=100_000) / 100.0  # Convert score to centipawns

    # Return None if the evaluation is too extreme (checkmate or stalemate)
    if abs(score) > 100:
        return None
    return score


def _get_file_count(pgn_file: str, output_dir: str) -> int:
    """
    Returns the next available file count for a given PGN file name.
    Ensures filenames follow the pattern: pgn_file_1_1000.txt, pgn_file_2_1000.txt, etc.
    """
    pattern = re.compile(rf"^{re.escape(pgn_file)}_(\d+)_\d+\.txt$")
    max_count = 0

    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            file_count = int(match.group(1))
            max_count = max(max_count, file_count)

    return max_count + 1


def _get_filename_without_extension(file_path: str) -> str:
    """Extracts the filename without the extension."""
    return os.path.splitext(os.path.basename(file_path))[0]


def parse_and_evaluate_positions(pgn_file, output_dir, max_positions_per_file=1000):
    """
    Parses a PGN file, extracts FEN positions, evaluates them using Stockfish,
    and saves them to text files in the specified output directory.
    Skips over positions already saved based on `_get_file_count`.
    """
    file_name = _get_filename_without_extension(pgn_file)
    file_count = _get_file_count(file_name, output_dir)
    skipped_positions = file_count * max_positions_per_file  # Calculate how many positions to skip
    current_position = 0  # Track the current position number while reading the PGN

    positions = []
    evaluations = []
    position_count = 0

    try:
        with open(pgn_file, 'r') as pgn:
            game = chess.pgn.read_game(pgn)

            while game:
                board = game.board()

                current_position += 1

                # Skip positions until reaching the starting point
                if current_position <= skipped_positions:
                    continue

                # Iterate over each move in the game
                for move in game.mainline_moves():
                    board.push(move)
                    fen = board.fen()
                    eval_score = _get_stockfish_evaluation(board)

                    if eval_score is None:
                        continue  # Skip positions with invalid evaluation (e.g., checkmate)


                    positions.append(fen)
                    evaluations.append(eval_score)
                    position_count += 1

                    # Save positions to a new file if the max is reached
                    if position_count >= max_positions_per_file:
                        output_file = os.path.join(output_dir, f"{file_name}_{file_count}_{max_positions_per_file}.txt")
                        with open(output_file, "w") as f:
                            for fen, eval_score in zip(positions, evaluations):
                                f.write(f"{fen},{eval_score}\n")
                        print(f"Saved {position_count} positions to {output_file}")

                        # Reset for the next file
                        positions.clear()
                        evaluations.clear()
                        position_count = 0
                        file_count += 1

                # Read the next game from the PGN file
                game = chess.pgn.read_game(pgn)

        # Save any remaining positions in the last file
        if positions:
            output_file = os.path.join(output_dir, f"{file_name}_{file_count}_{max_positions_per_file}.txt")
            with open(output_file, "w") as f:
                for fen, eval_score in zip(positions, evaluations):
                    f.write(f"{fen},{eval_score}\n")
            print(f"Saved {position_count} positions to {output_file}")
    except FileNotFoundError:
        print(f"Error: The PGN file {pgn_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # Parse PGN file and evaluate positions
    parse_and_evaluate_positions(PGN_PATH, OUTPUT_DIRECTORY, max_positions_per_file=100)

    # Close Stockfish engine after processing
    engine.close()

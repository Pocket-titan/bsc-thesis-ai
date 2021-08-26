from .dataloader import create_dataloader
from .datasets import large_df, medium_df, simple_df, small_df
from .pgm import generate_data

simple_dataloader = create_dataloader(simple_df)
small_dataloader = create_dataloader(small_df)
medium_dataloader = create_dataloader(medium_df)
large_dataloader = create_dataloader(large_df)

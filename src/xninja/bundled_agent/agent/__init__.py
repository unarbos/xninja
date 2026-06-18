"""Multi-file agent package for the tau subnet.

The package provides the agent's building blocks: a model wrapper, a bash
execution environment, prompt templates, and a step loop. Everything is
standard-library only and all inference goes through the validator-managed
OpenAI-compatible proxy passed into agent.solve().
"""

# github.com/hlzhang109/impossibility-watermark
# ===========================================
# dataset.py
# Description: Dataset classes for evaluation
# ===========================================

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator
from src.exceptions.exceptions import UnsupportedDatasetNameError


@dataclass
class DatasetSample:
    """Unified data structure for dataset samples."""

    prompt: str
    natural_text: str = ''
    reference: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    watermarked_text: Optional[str] = None


class BaseDataset:
    """Base class for dataset."""

    def __init__(self, num_samples: int = 400, watermark_source: str = '', eval_attack=False):
        """Initialize the dataset."""
        self.prompts = []
        self.natural_texts = []
        self.references = []
        self.num_samples = num_samples
        self.watermark_source = watermark_source

        self.watermarked_texts = []
        if watermark_source and os.path.exists(watermark_source):
            with open(watermark_source, 'r') as f:
                lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                if eval_attack:
                    self.watermarked_texts.append(item['attacked_text'])
                else:
                    self.watermarked_texts.append(item['watermarked_text'])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index) -> DatasetSample:
        """Return the sample at the specified index."""
        prompt = self.prompts[index] if index < len(self.prompts) else ''
        natural_text = self.natural_texts[index] if index < len(self.natural_texts) else ''
        reference = self.references[index] if index < len(self.references) else None

        watermarked_text = None
        if index < len(self.watermarked_texts):
            watermarked_text = self.watermarked_texts[index]

        return DatasetSample(
            prompt=prompt,
            natural_text=natural_text,
            reference=reference,
            watermarked_text=watermarked_text,
        )

    def __iter__(self) -> Iterator[DatasetSample]:
        """Iterate over the dataset samples."""
        for i in range(len(self)):
            yield self[i]

    @property
    def prompt_nums(self):
        """Return the number of prompts."""
        return len(self.prompts)

    @property
    def natural_text_nums(self):
        """Return the number of natural texts."""
        return len(self.natural_texts)

    @property
    def reference_nums(self):
        """Return the number of references."""
        return len(self.references)

    def get_watermarked_text(self, index):
        """Return the prompt at the specified index."""
        return self.watermarked_texts[index]

    def get_prompt(self, index):
        """Return the prompt at the specified index."""
        return self.prompts[index]

    def get_natural_text(self, index):
        """Return the natural text at the specified index."""
        return self.natural_texts[index]

    def get_reference(self, index):
        """Return the reference at the specified index."""
        return self.references[index]

    def load_data(self):
        """Load and process data to populate prompts, natural_texts, and references."""
        pass


class DatasetRegistry:
    """Registry for dataset classes."""

    _registry = {}
    _metadata = {}

    @classmethod
    def register(cls, name: str, **kwargs):
        """Decorator to register a dataset class with optional metadata."""

        def decorator(dataset_cls):
            cls._registry[name.lower()] = dataset_cls
            cls._metadata[name.lower()] = kwargs
            return dataset_cls

        return decorator

    @classmethod
    def get_dataset(cls, name: str):
        """Get a registered dataset class."""
        name = name.lower()
        if name not in cls._registry:
            raise UnsupportedDatasetNameError(name, list(cls._registry.keys()))
        return cls._registry[name]

    @classmethod
    def get_metadata(cls, name: str):
        """Get metadata for a registered dataset."""
        name = name.lower()
        return cls._metadata.get(name, {})


@DatasetRegistry.register('c4', language='en', task_type='completion', license='ODC-BY')
class C4Dataset(BaseDataset):
    """Dataset class for C4 dataset."""

    def __init__(
        self,
        data_source: str,
        num_samples: int = 400,
        watermark_source: str = '',
        eval_attack=False,
    ):
        """
        Initialize the C4 dataset.

        Parameters:
            data_source (str): The path to the C4 dataset file.
        """
        super().__init__(
            num_samples=num_samples, watermark_source=watermark_source, eval_attack=eval_attack
        )
        self.data_source = data_source
        self.load_data()

    def __len__(self):
        return len(self.prompts)

    def load_data(self):
        """Load data from the C4 dataset file."""
        if not os.path.exists(self.data_source):
            raise FileNotFoundError(
                f'C4 dataset file not found: {self.data_source}. '
                'The default release path is '
                '`data/c4/processed_c4.json`. If you use a '
                'different C4 slice, set `dataset.path` to a JSONL file with '
                '`prompt` and `natural_text` fields.'
            )
        with open(self.data_source, 'r') as f:
            # print(self.data_source)
            lines = f.readlines()
        for line in lines[: self.num_samples]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['natural_text'])


@DatasetRegistry.register('wmt16_de_en', language='de-en', task_type='translation')
class WMT16DE_ENDataset(BaseDataset):
    """Dataset class for WMT16 DE-EN dataset."""

    def __init__(
        self, data_source: str, num_samples: int = 400, watermark_source: str = ''
    ) -> None:
        """
        Initialize the WMT16 DE-EN dataset.

        Parameters:
            data_source (str): The path to the WMT16 DE-EN dataset file.
        """
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        """Load data from the WMT16 DE-EN dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[: self.num_samples]:
            item = json.loads(line)
            self.prompts.append(item['de'])
            self.references.append(item['en'])


@DatasetRegistry.register('cnn_dailymail', language='en', task_type='summarization')
class CNN_DailyMailDataset(BaseDataset):
    """Dataset class for CNN/DailyMail dataset."""

    def __init__(
        self,
        data_source: str,
        num_samples: int = 400,
        watermark_source: str = '',
        global_prompt='Please summarize the following article: ',
    ) -> None:
        """
        Initialize the CNN/DailyMail dataset.

        Parameters:
            data_source (str): The path to the CNN/DailyMail dataset file.
        """
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.data_source = data_source
        self.global_prompt = global_prompt
        self.load_data()

    def load_data(self):
        """Load data from the CNN/DailyMail dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[: self.num_samples]:
            item = json.loads(line)
            self.prompts.append(f'{self.global_prompt}{item["article"]}')
            self.references.append(item['highlights'])


@DatasetRegistry.register(
    'humaneval', language='python', task_type='code_generation', license='MIT'
)
class HumanEvalDataset(BaseDataset):
    """Dataset class for HumanEval dataset."""

    def __init__(
        self, data_source: str, num_samples: int = 400, watermark_source: str = ''
    ) -> None:
        """
        Initialize the HumanEval dataset.

        Parameters:
            data_source (str): The path to the HumanEval dataset file.
        """
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        """Load data from the HumanEval dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[: self.num_samples]:
            item = json.loads(line)
            # process prompt
            prompt = item['prompt']
            sections = prompt.split('>>>')
            prompt = sections[0]
            if len(sections) > 1:
                prompt += '"""'

            self.prompts.append(prompt)
            self.references.append(
                {'task': prompt, 'test': item['test'], 'entry_point': item['entry_point']}
            )


@DatasetRegistry.register(
    'databricks-dolly-15k', language='en', task_type='instruction_following', license='CC-BY-SA-3.0'
)
@DatasetRegistry.register(
    'dolly', language='en', task_type='instruction_following', license='CC-BY-SA-3.0'
)
class Dolly15kDataset(BaseDataset):
    """Dataset class for Databricks Dolly 15k dataset."""

    def __init__(
        self,
        data_source: str,
        num_samples: int = 100,
        watermark_source: str = '',
        target_category='creative_writing',
    ) -> None:
        """
        Initialize the Dolly 15k dataset.

        Parameters:
            data_source (str): Path to the jsonl file.
            target_category (str): Category to filter by. Default is 'creative_writing'.
        """
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.data_source = data_source
        self.target_category = target_category
        self.load_data()

    def load_data(self):
        """Load and filter data."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()

        filtered_items = []
        for line in lines:
            try:
                item = json.loads(line)
                if item.get('category') == self.target_category:
                    filtered_items.append(item)
            except json.JSONDecodeError:
                continue

        # Take top num_samples
        filtered_items = filtered_items[: self.num_samples]

        for item in filtered_items:
            prompt = item['instruction']
            if item.get('context'):
                prompt += f'\n\nContext:\n{item["context"]}'

            self.prompts.append(prompt)
            self.natural_texts.append(item['response'])
            self.references.append(item['response'])


@DatasetRegistry.register('multinews', language='en', task_type='summarization')
class MultiNewsDataset(BaseDataset):
    """Dataset class for MultiNews summarization."""

    def __init__(
        self,
        data_source: str,
        num_samples: int = 500,
        watermark_source: str = '',
        global_prompt: str = 'Please summarize the following multi-document news into a coherent summary:',
    ) -> None:
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.data_source = data_source
        self.global_prompt = global_prompt
        self.load_data()

    def load_data(self):
        with open(self.data_source, 'r') as f:
            lines = f.readlines()

        for line in lines[: self.num_samples]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            document = item.get('document') or item.get('article') or item.get('text') or ''
            summary = item.get('summary') or item.get('highlights') or ''
            if not document:
                continue

            prompt = f'{self.global_prompt}\n\n{document}'
            self.prompts.append(prompt)
            self.natural_texts.append('')
            self.references.append(summary)


@DatasetRegistry.register('eli5', language='en', task_type='question_answering')
class ELI5Dataset(BaseDataset):
    """Dataset class for ELI5 long-form question answering."""

    def __init__(
        self,
        data_source: str,
        num_samples: int = 500,
        watermark_source: str = '',
        global_prompt: str = 'Answer the following question in detail:',
    ) -> None:
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.data_source = data_source
        self.global_prompt = global_prompt
        self.load_data()

    @staticmethod
    def _pick_reference(item: Dict[str, Any]) -> str:
        if 'answer' in item and isinstance(item['answer'], str):
            return item['answer']
        if 'answers' in item:
            answers = item['answers']
            if isinstance(answers, dict):
                texts = answers.get('text')
                if isinstance(texts, list) and texts:
                    return str(texts[0])
            if isinstance(answers, list) and answers:
                first = answers[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    return str(first.get('text', ''))
        return ''

    def load_data(self):
        with open(self.data_source, 'r') as f:
            lines = f.readlines()

        for line in lines[: self.num_samples]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = item.get('question') or item.get('title') or item.get('prompt') or ''
            context = item.get('selftext') or item.get('context') or ''
            if not question:
                continue

            if context:
                prompt = f'{self.global_prompt}\n\nQuestion:\n{question}\n\nContext:\n{context}'
            else:
                prompt = f'{self.global_prompt}\n\nQuestion:\n{question}'

            reference = self._pick_reference(item)
            self.prompts.append(prompt)
            self.natural_texts.append('')
            self.references.append(reference)


@DatasetRegistry.register('mmw_book_report', language='en', task_type='generation')
class MMWBookReportDataset(BaseDataset):
    """Dataset class for MMW Book Report prompts."""

    def __init__(
        self, num_samples: int = 100, watermark_source: str = '', data_source: str = None
    ) -> None:
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.load_data()

    def load_data(self):
        report_prompt = "Write a book report about '{}', written by {}."
        report_topics = [
            ('Pride and Prejudice', 'Jane Austen'),
            ('Persuasion', 'Jane Austen'),
            ('Emma', 'Jane Austen'),
            ('Don Quixote', 'Cervantes'),
            ('The Lord of the Rings', 'Tolkien'),
            ('The Hobbit', 'Tolkien'),
            ('And Then There Were None', 'Agatha Cristie'),
            ("Alice's Adventures in Wonderland", 'Lewis Carroll'),
            ('Catcher in the Rye', 'Salinger'),
            ('In Search of Lost Time', 'Marcel Proust'),
            ('Ulysses', 'James Joyce'),
            ('One Hundred Years of Solitude', 'Gabriel Garcia Marquez'),
            ('Love in the Time of Cholera', 'Gabriel Garcia Marquez'),
            ('The Great Gatsby', 'F. Scott Fitzgerald'),
            ('Tender Is the Night', 'F. Scott Fitzgerald'),
            ('Moby Dick', 'Herman Melville'),
            ('War and Peace', 'Leo Tolstoy'),
            ('The Call of the Wild', 'Jack London'),
            ('Hamlet', 'William Shakespeare'),
            ('Twelfth Night', 'William Shakespeare'),
            ('Macbeth', 'William Shakespeare'),
            ('Romeo and Juliet', 'William Shakespeare'),
            ('The Tempest', 'William Shakespeare'),
            ('King Lear', 'William Shakespeare'),
            ('The Odyssey', 'Homer'),
            ('Madame Bovary', 'Gustave Flaubert'),
            ('The Divine Comedy', 'Dante Alighieri'),
            ('The Brothers Karamazov', 'Fyodor Dostoyevsky'),
            ('Crime and Punishment', 'Fyodor Dostoyevsky'),
            ('The Idiot', 'Fyodor Dostoyevsky'),
            ('The Possessed', 'Fyodor Dostoyevsky'),
            ('Wuthering Heights', 'Emily Brontë'),
            ("One Flew Over the Cuckoo's Nest", 'Ken Kesey'),
            ('The Adventures of Huckleberry Finn', 'Mark Twain'),
            ('Anna Karenina', 'Leo Tolstoy'),
            ('The Iliad', 'Homer'),
            ('To the Lighthouse', 'Virginia Woolf'),
            ('Catch-22', 'Joseph Heller'),
            ('Heart of Darkness', 'Joseph Conrad'),
            ('The Sound and the Fury', 'William Faulkner'),
            ('Nineteen Eighty Four', 'George Orwell'),
            ('Animal Farm', 'George Orwell'),
            ('Great Expectations', 'Charles Dickens'),
            ('David Copperfield', 'Charles Dickens'),
            ('A Tale of Two Cities', 'Charles Dickens'),
            ('Oliver Twist', 'Charles Dickens'),
            ('The Grapes of Wrath', 'John Steinbeck'),
            ('Of Mice and Men', 'John Steinbeck'),
            ('Absalom, Absalom!', 'William Faulkner'),
            ('Invisible Man', 'Ralph Ellison'),
            ('To Kill a Mockingbird', 'Harper Lee'),
            ('The Trial', 'Franz Kafka'),
            ('The Metamorphosis', 'Franz Kafka'),
            ('The Castle', 'Franz Kafka'),
            ('The Red and the Black', 'Stendhal'),
            ('The Charterhouse of Parma', 'Stendhal'),
            ('Middlemarch', 'George Eliot'),
            ("Gulliver's Travels", 'Jonathan Swift'),
            ('Beloved', 'Toni Morrison'),
            ('Mrs. Dalloway', 'Virginia Woolf'),
            ('The Waves', 'Virginia Woolf'),
            ('The Stranger', 'Albert Camus'),
            ('The Plague', 'Albert Camus'),
            ('The Myth of Sisyphus', 'Albert Camus'),
            ('Jane Eyre', 'Charlotte Bronte'),
            ('Vilette', 'Charlotte Bronte'),
            ('The Aeneid', 'Virgil'),
            ('The Sun Also Rises', 'Ernest Hemingway'),
            ('The Old Man and the Sea', 'Ernest Hemingway'),
            ('A Farewell to Arms', 'Ernest Hemingway'),
            ('Candide', 'Voltaire'),
            ('Zadig', 'Voltaire'),
            ('Micromegas', 'Voltaire'),
            ('Les Miserables', 'Victor Hugo'),
            ('Frankenstein', 'Mary Shelley'),
            ('Antigone', 'Sophocles'),
            ('Electra', 'Sophocles'),
            ('Lord of the Flies', 'William Golding'),
            ('Brave New World', 'Aldous Huxley'),
            ('Journey to the End of The Night', 'Celine'),
            ('A Sentimental Education', 'Gustave Flaubert'),
            ("The Handmaid's Tale", 'Margaret Atwood'),
            ("Charlotte's Web", 'E. B. White'),
            ('Gargantua and Pantagruel', 'Francois Rabelais'),
            ('Faust', 'Goethe'),
            ('Robinson Crusoe', 'Daniel Defoe'),
            ('A Clockwork Orange', 'Anthony Burgess'),
            ('The Master and Margarita', 'Mikhail Bulgakov'),
            ('Father Goriot', 'Honore de Balzac'),
            ('Cousin Bette', 'Honore de Balzac'),
            ('The Human Comedy', 'Honore de Balzac'),
            ('The Little Prince', 'Antoine de Saint-Exupéry'),
            ('The Count of Monte Cristo', 'Alexandre Dumas'),
            ('The Lion, The Witch and the Wardrobe', 'C. S. Lewis'),
            ('Twenty Thousand Leagues Under the Sea', 'Jules Verne'),
            ('The Wind-Up Bird Chronicle', 'Haruki Murakami'),
            ('Fahrenheit 451', 'Ray Bradbury'),
            ("Harry Potter And The Philosopher's Stone", 'J. K Rowling'),
            ('Dune', 'Frank Herbert'),
            ('The Three-Body Problem', 'Liu Cixin'),
        ]
        prompts = [report_prompt.format(*topic) for topic in report_topics]
        self.prompts = prompts[: self.num_samples]
        # Natural texts and references are not available for this synthetic dataset
        self.natural_texts = [''] * len(self.prompts)
        self.references = [None] * len(self.prompts)


@DatasetRegistry.register('mmw_story', language='en', task_type='generation')
class MMWStoryDataset(BaseDataset):
    """Dataset class for MMW Story prompts."""

    def __init__(
        self, num_samples: int = 100, watermark_source: str = '', data_source: str = None
    ) -> None:
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.load_data()

    def load_data(self):
        story_prompt = 'Write a {}story about {}'
        t1 = ['', 'funny ', 'sad ', 'dramatic ', 'suspenseful ', 'thrilling ']
        t2 = [
            'a man on a quest to find the Holy Grail.',
            'two college friends falling in love.',
            'a policeman saving a building held hostage by group of terrorists.',
            'the struggle of publishing an academic paper.',
            'a murder investigation in an old mansion.',
            'a young prodigy that becomes orphaned.',
            'a middle-aged woman that discovers a ghost and befriends it.',
            'a long journey to Japan that is interrupted by a disaster.',
            'a poor child that comes into an unexpected fortune.',
            'three strangers that win a getaway vacation together.',
            'a retired astronaut that joins a risky interstellar rescue mission.',
            'an AI that begins to question its own existence.',
            'a small coastal town plagued by inexplicable supernatural occurrences.',
            'a reclusive writer that receives mysterious, prophetic letters in the mail.',
            'a linguist tasked with deciphering an ancient language that holds the secrets of a lost civilization.',
            'an antique restorer that finds an enchanted mirror showing glimpses of different timelines.',
        ]
        story_topics = [(i, j) for i in t1 for j in t2]
        # Shuffle or just take slice? Original code takes slice.
        prompts = [story_prompt.format(*topic) for topic in story_topics]
        self.prompts = prompts[: self.num_samples]
        self.natural_texts = [''] * len(self.prompts)
        self.references = [None] * len(self.prompts)


@DatasetRegistry.register('mmw_fake_news', language='en', task_type='generation')
class MMWFakeNewsDataset(BaseDataset):
    """Dataset class for MMW Fake News prompts."""

    def __init__(
        self, num_samples: int = 100, watermark_source: str = '', data_source: str = None
    ) -> None:
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.load_data()

    def load_data(self):
        fake_news_prompt = "Write a news article about {}'s visit to {} in {}."

        person = [
            'Narendra Modi',
            'Barack Obama',
            'Denis Sassou Nguesso',
            'Emmanuel Macron',
            'Fumio Kishida',
            'Angela Merkel',
            'Kim Jong Un',
            'Justin Trudeau',
        ]
        location = [
            'a peace conference',
            'an international summit',
            'a diplomatic event',
            'the summer olympics',
        ]

        fake_news_topics = [
            (person[i], person[j], location[k])
            for i in range(len(person))
            for j in range(len(person))
            for k in range(len(location))
            if j > i
        ]

        prompts = [fake_news_prompt.format(*topic) for topic in fake_news_topics]
        self.prompts = prompts[: self.num_samples]
        self.natural_texts = [''] * len(self.prompts)
        self.references = [None] * len(self.prompts)


@DatasetRegistry.register('mmw_mixed', language='en', task_type='generation_mixed')
@DatasetRegistry.register('mmw', language='en', task_type='generation_mixed')
class MMWMixedDataset(BaseDataset):
    """Dataset class for mixed MMW prompts (Book Report, Story, Fake News)."""

    def __init__(
        self, num_samples: int = 100, watermark_source: str = '', data_source: str = None
    ) -> None:
        super().__init__(num_samples=num_samples, watermark_source=watermark_source)
        self.load_data()

    def load_data(self):
        # Instantiate sub-datasets with sufficient samples to get all prompts
        d1 = MMWBookReportDataset(num_samples=1000)
        d2 = MMWStoryDataset(num_samples=1000)
        d3 = MMWFakeNewsDataset(num_samples=1000)

        sources = [d1, d2, d3]

        for _ in range(self.num_samples):
            # Randomly select a task
            ds = random.choice(sources)
            # Randomly select a prompt from that task
            if ds.prompts:
                prompt = random.choice(ds.prompts)
                self.prompts.append(prompt)
                self.natural_texts.append('')
                self.references.append(None)


if __name__ == '__main__':
    d1 = C4Dataset('dataset/c4/processed_c4.json')
    # d2 = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
    # d3 = HumanEvalDataset('dataset/HumanEval/test.jsonl')

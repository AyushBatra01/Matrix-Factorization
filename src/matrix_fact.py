import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

class MatrixFact:
    def __init__(self, matrix, K, ratio=5, eta=0.01, lambda_=0.01,
                 alpha=0.2, c_neg=1.0, max_iter=100, val_ratio=0.1,
                 random_seed=None):
        """
        Class for matrix factorization object

        Parameters
        ----------
        matrix: scipy.sparse.scr_matrix of user-recipe interactions
                users along rows, recipes along columns
                entries are integers where 0 indicates no interaction and interactions have rating values 1-5

        K: latent dimension size

        ratio: negative to positive sample ratio

        eta: initial learning rate

        lambda_: regularization strength

        alpha: confidence multiplier for higher ratings
                confidence value will be 1 + alpha * rating

        c_neg: confidence multiplier for no interactions
                confidence value will be c_neg

        max_iter: maximum training epochs

        val_ratio: ratio of positive samples to be used for validation

        random_seed: random initialization for reproducibility
        """

        self.n, self.m = matrix.shape
        self.matrix = matrix
        self.K = K
        self.ratio = ratio
        self.eta = eta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.c_neg = c_neg
        self.max_iter = max_iter
        self.val_ratio = val_ratio
        self.results = None

        if random_seed is not None:
            np.random.seed(random_seed)

        # initializations
        self.U = np.random.normal(size=(self.n, K), scale=1/np.sqrt(K)).astype(np.float32)
        self.V = np.random.normal(size=(self.m, K), scale=1/np.sqrt(K)).astype(np.float32)

        self.rng = np.random.default_rng()

        # store train and validation sets
        self.train_matrix, self.val_matrix = self._create_train_val_split()

    
    def _create_train_val_split(self):
        """
        Randomly choose items for validation set
        """
        train_data, train_rows, train_cols = [], [], []
        val_data, val_rows, val_cols = [], [], []

        for u in range(self.n):
            pos_items = self.matrix[u].indices
            pos_ratings = self.matrix[u].data

            if len(pos_items) == 0:
                continue

            # randomly choose validation samples
            n_val = int(len(pos_items) * self.val_ratio)
            val_idx = self.rng.choice(len(pos_items), size=n_val, replace=False)

            train_mask = np.ones(len(pos_items), dtype=bool)
            train_mask[val_idx] = False

            train_items = pos_items[train_mask]
            train_ratings = pos_ratings[train_mask]

            val_items = pos_items[val_idx]
            val_ratings = pos_ratings[val_idx]

            train_rows.extend([u] * len(train_items))
            train_cols.extend(train_items)
            train_data.extend(train_ratings)

            val_rows.extend([u] * len(val_items))
            val_cols.extend(val_items)
            val_data.extend(val_ratings)

        # store as sparse matrices
        train_matrix = csr_matrix((train_data, (train_rows, train_cols)), shape=(self.n, self.m))
        val_matrix = csr_matrix((val_data, (val_rows, val_cols)), shape=(self.n, self.m))

        return train_matrix, val_matrix

    
    def sigmoid(self, x):
        """
        Sigmoid function
        """
        x = np.clip(x, -20, 20)
        return 1 / (1 + np.exp(-x))

    
    def step_decay(self, eta, epoch, rate=0.5, interval=10):
        """
        Decay learning rate geometrically
        """
        return eta * (rate ** (epoch // interval))


    def sample_negatives(self, u, n):
        """
        Randomly sample n negative items for user u
        """
        pos_items = set(self.train_matrix[u].indices)
        negs = []
        while len(negs) < n:
            i = self.rng.integers(0, self.m)
            if i not in pos_items:
                negs.append(i)
        return np.array(negs)

    
    def forward_pass(self, u, matrix):
        """
        Compute predictions for a forward pass for user u on given matrix

        Returns (idx, y, pred, c) where
            idx: index of recipes predicted for
            y: ground truth indicators of an interaction (0 or 1)
            pred: prediction (between 0 and 1)
            c: confidence weights based on y values
        """
        pos_items = matrix[u].indices
        pos_ratings = matrix[u].data

        if len(pos_items) == 0:
            return None

        # sample negatives
        n_neg = self.ratio * len(pos_items)
        neg_items = self.sample_negatives(u, n_neg)

        idx = np.concatenate([pos_items, neg_items])
        y = np.concatenate([np.ones(len(pos_items)), np.zeros(len(neg_items))])

        # weights
        pos_wt = 1 + self.alpha * pos_ratings
        neg_wt = self.c_neg * np.ones(len(neg_items))
        c = np.concatenate([pos_wt, neg_wt])

        # compute predicitons
        V_sub = self.V[idx]
        logits = V_sub @ self.U[u]
        pred = self.sigmoid(logits)

        return idx, y, pred, c

    
    def neg_log_loss(self, y, pred, c):
        """
        Compute negative log loss on a vector of ground truth, predictions, and confidence values

        Loss is scaled proportional to confidence values

        Returns vector of same size of inputs
        """
        eps = 1e-8
        pred = np.clip(pred, eps, 1 - eps)
        return -c * (y * np.log(pred) + (1 - y) * np.log(1 - pred))

    
    def loss(self, matrix, prop_users=0.1):
        """
        Compute loss for a random sample of users
        """
        # get users
        if prop_users == 1:
            users = np.arange(self.n) 
        else:
            users = self.rng.choice(self.n, size=max(1, int(prop_users * self.n)), replace=False)
    
        total_loss = 0.0
        pos_loss = 0.0
        neg_loss = 0.0
        count = 0
    
        for u in users:
            # forward pass
            res = self.forward_pass(u, matrix)
            if res is None:
                continue
            _, y, pred, c = res

            # calculate loss
            losses = self.neg_log_loss(y, pred, c)
    
            # masks
            pos_mask = (y == 1)
            neg_mask = (y == 0)
    
            # store positive/negative losses
            if np.any(pos_mask):
                pos_loss += np.mean(losses[pos_mask])
            if np.any(neg_mask):
                neg_loss += np.mean(losses[neg_mask])

            # store total loss
            total_loss += np.mean(losses)
            count += 1
    
        if count == 0:
            return 0.0, 0.0, 0.0
    
        return (total_loss / count, pos_loss / count, neg_loss / count)

    
    def user_step(self, u, lr, max_grad_norm=5):
        """
        Take a gradient step for user u, clip gradient norm at max_grad_norm if exceeds it
        """
        # forward pass
        res = self.forward_pass(u, self.train_matrix)
        if res is None:
            return
        idx, y, pred, c = res

        # error
        err = c * (pred - y)

        # get subset of recipe latent vectors
        V_sub = self.V[idx]

        # calculate gradients
        grad_U = err @ V_sub
        grad_V = err[:, None] * self.U[u][None, :]

        # clip
        norm = np.linalg.norm(grad_U)
        if norm > max_grad_norm:
            grad_U *= max_grad_norm / norm

        # update
        self.U[u] -= lr * (grad_U + self.lambda_ * self.U[u])
        self.V[idx] -= lr * (grad_V + self.lambda_ * self.V[idx])

    
    def normalize_embeddings(self):
        """
        Normalize user and recipe embeddings to unit norm
        """
        self.U /= (np.linalg.norm(self.U, axis=1, keepdims=True) + 1e-8)
        self.V /= (np.linalg.norm(self.V, axis=1, keepdims=True) + 1e-8)

    
    def fit(self, normalize_every=5, progress=True):
        """
        Train the matrix factorization model

        progress=True will show a tqdm progress bar
        """
        train_losses, train_pos_losses, train_neg_losses = [], [], []
        val_losses, val_pos_losses, val_neg_losses = [], [], []
        best_val_loss = 99999

        for epoch in tqdm(range(self.max_iter), disable=not progress):
            # decay learning rate
            lr = self.step_decay(self.eta, epoch)

            # take gradient steps
            for u in range(self.n):
                self.user_step(u, lr)

            # normalize embeddings if necessary
            if normalize_every and (epoch + 1) % normalize_every == 0:
                self.normalize_embeddings()

            # calculate losses
            train_total, train_pos, train_neg = self.loss(self.train_matrix, prop_users=0.2)
            val_total, val_pos, val_neg = self.loss(self.val_matrix, prop_users=0.2)

            # store losses
            train_losses.append(train_total)
            val_losses.append(val_total)
            train_pos_losses.append(train_pos)
            train_neg_losses.append(train_neg)
            val_pos_losses.append(val_pos)
            val_neg_losses.append(val_neg)

            # keep track of best results on validation set
            if val_total < best_val_loss:
                best_val_loss = vl
                best_U = self.U.copy()
                best_V = self.V.copy()

        # set to best validation results
        self.U, self.V = best_U, best_V

        # store results
        self.results = {
            "train": {
                "total": train_losses,
                "pos": train_pos_losses,
                "neg": train_neg_losses
            },
            "val": {
                "total": val_losses,
                "pos": val_pos_losses,
                "neg": val_neg_losses
            }
        }

        return self.U, self.V, self.results

    
    def predict_existing_user(self, user_id, top_k=10, exclude_liked=True):
        """
        Predict top recipes for an existing user
        """
        # compute scores for user
        scores = self.sigmoid(self.V @ self.U[user_id])

        # remove recipes with existing interactions
        if exclude_liked:
            liked = self.matrix[user_id].indices
            scores[liked] = -np.inf

        # get top results
        top_idx = np.argsort(scores)[::-1][:top_k]
        return scores[top_idx], top_idx

    
    def new_user_vector(self, recipe_indices, recipe_ratings=None):
        """
        Estimate vector for a new user by average of liked item vectors

        recipe_indices: list of reipces the user has interacted with

        recipe_ratings: list of ratings for the recipes the user has interacted with
        """
        # return mean embedding if no information
        if len(recipe_indices) == 0:
            return np.mean(self.U, axis=0)

        recipe_indices = np.array(recipe_indices)
        if recipe_ratings is None:
            recipe_ratings = np.ones(len(recipe_indices))

        # take average of liked item vectors
        V_liked = self.V[recipe_indices]
        weights = recipe_ratings / np.sum(recipe_ratings)
        return np.sum(V_liked * weights[:, None], axis=0)

    
    def predict_new_user(self, liked_items, liked_ratings=None, top_k=10):
        """
        Predict top recipe recommendations for a new user

        liked_items: list indices for recipes the user has interacted with

        liked_ratings: list of ratings for the recipes the user has interacted with

        top_k: number of recommendations to return
        """
        # estimate new user's vector
        user_vec = self.new_user_vector(liked_items, liked_ratings)

        # compute scores
        scores = self.sigmoid(self.V @ user_vec)
        scores[liked_items] = -np.inf

        # get top results
        top_idx = np.argsort(scores)[::-1][:top_k]
        return scores[top_idx], top_idx